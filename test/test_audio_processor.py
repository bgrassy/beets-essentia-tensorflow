"""Tests for updated audio processing functionality."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections.abc import Generator
from unittest.mock import MagicMock, patch
import json
import numpy as np

import pytest
from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs

from beetsplug.essentia_tensorflow.audio_processor import (
    AudioProcessor,
    AudioProcessingError,
    FeatureExtractionResult,
)
from beetsplug.essentia_tensorflow.model_manager import ModelManager, ModelConfiguration


class TestFeatureExtractionResult:
    """Tests for the FeatureExtractionResult class (unchanged from original)."""

    def test_initialization(self) -> None:
        """Test initialization with different configurations."""
        # Test with complete configuration
        result = FeatureExtractionResult(
            feature_type="genre",
            model_name="genre_model",
            values=[0.8, 0.1, 0.1],
            probabilities=[0.8, 0.1, 0.1],
            labels=["rock", "pop", "jazz"],
            confidence=0.8,
        )
        assert result.feature_type == "genre"
        assert result.model_name == "genre_model"
        assert result.values == [0.8, 0.1, 0.1]
        assert result.probabilities == [0.8, 0.1, 0.1]
        assert result.labels == ["rock", "pop", "jazz"]
        assert result.confidence == 0.8

        # Test with minimal configuration
        result = FeatureExtractionResult(
            feature_type="bpm",
            model_name="tempo_model",
            values=[120.5],
        )
        assert result.feature_type == "bpm"
        assert result.model_name == "tempo_model"
        assert result.values == [120.5]
        assert result.probabilities is None
        assert result.labels is None
        assert result.confidence is None

    def test_get_top_result(self) -> None:
        """Test getting top result."""
        result = FeatureExtractionResult(
            feature_type="genre",
            model_name="genre_model",
            values=[0.8, 0.1, 0.1],
            probabilities=[0.1, 0.8, 0.1],
            labels=["rock", "pop", "jazz"],
        )

        top_label, top_prob = result.get_top_result()
        assert top_label == "pop"
        assert top_prob == 0.8

        # Test with invalid configuration
        result = FeatureExtractionResult(
            feature_type="bpm",
            model_name="tempo_model",
            values=[120.5],
        )
        with pytest.raises(ValueError):
            result.get_top_result()

    def test_get_top_n_results(self) -> None:
        """Test getting top N results."""
        result = FeatureExtractionResult(
            feature_type="genre",
            model_name="genre_model",
            values=[0.1, 0.3, 0.2, 0.4],
            probabilities=[0.1, 0.3, 0.2, 0.4],
            labels=["rock", "pop", "jazz", "classical"],
        )

        top_results = result.get_top_n_results(2)
        assert len(top_results) == 2
        assert top_results[0][0] == "classical"
        assert top_results[0][1] == 0.4
        assert top_results[1][0] == "pop"
        assert top_results[1][1] == 0.3

        # Test with invalid configuration
        result = FeatureExtractionResult(
            feature_type="bpm",
            model_name="tempo_model",
            values=[120.5],
        )
        with pytest.raises(ValueError):
            result.get_top_n_results()

    def test_format_for_database(self) -> None:
        """Test formatting results for database storage."""
        # Test classification result
        result = FeatureExtractionResult(
            feature_type="genre",
            model_name="genre_model",
            values=[0.1, 0.8, 0.1],
            probabilities=[0.1, 0.8, 0.1],
            labels=["rock", "pop", "jazz"],
        )

        db_format = result.format_for_database()
        assert db_format["feature_type"] == "genre"
        assert db_format["model_name"] == "genre_model"
        assert db_format["value"] == "pop"
        assert db_format["confidence"] == 0.8
        assert db_format["probabilities"] == [0.1, 0.8, 0.1]

        # Test BPM result
        result = FeatureExtractionResult(
            feature_type="bpm",
            model_name="tempo_model",
            values=[120.5],
        )

        db_format = result.format_for_database()
        assert db_format["feature_type"] == "bpm"
        assert db_format["model_name"] == "tempo_model"
        assert db_format["value"] == 120.5

        # Test with metadata
        result = FeatureExtractionResult(
            feature_type="key",
            model_name="key_model",
            values=[0.1, 0.8, 0.1],
            probabilities=[0.1, 0.8, 0.1],
            labels=["C", "G", "D"],
            metadata={"segment_duration": 30},
        )

        db_format = result.format_for_database()
        assert db_format["feature_type"] == "key"
        assert db_format["value"] == "G"
        assert db_format["metadata"]["segment_duration"] == 30


class TestAudioProcessor:
    """Tests for the updated AudioProcessor class."""

    @pytest.fixture
    def mock_audio_data(self) -> np.ndarray:
        """Create mock audio data."""
        return np.random.random(44100 * 10)  # 10 seconds of random audio

    @pytest.fixture
    def tmp_path(self) -> Generator[Path, None, None]:
        """Create a temporary directory for test files."""
        tmp_dir = tempfile.mkdtemp()
        yield Path(tmp_dir)
        shutil.rmtree(tmp_dir)

    @pytest.fixture
    def create_metadata_files(self, tmp_path: Path) -> Dict[str, str]:
        """Create sample metadata JSON files."""
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()

        # Create sample metadata files
        discogs_metadata = {
            "name": "Discogs Embedding",
            "type": "Audio embedding",
            "model_path": str(tmp_path / "models/discogs_model.pb"),
            "schema": {"outputs": [{"name": "PartitionedCall:1"}]},
            "inference": {"sample_rate": 16000, "algorithm": "TensorflowPredictEffnetDiscogs"},
        }

        genre_metadata = {
            "name": "Genre Classifier",
            "type": "Music genre classification",
            "model_path": str(tmp_path / "models/genre_model.pb"),
            "classes": ["rock", "pop", "jazz", "classical"],
            "schema": {
                "inputs": [{"name": "serving_default_model_Placeholder"}],
                "outputs": [{"name": "PartitionedCall:0"}],
            },
            "inference": {
                "sample_rate": 16000,
                "algorithm": "TensorflowPredict2D",
                "embedding_model": {"model_name": "discogs"},
            },
        }

        key_metadata = {
            "name": "Key Detector",
            "type": "Music key detection",
            "model_path": str(tmp_path / "models/key_model.pb"),
            "classes": [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
                "Cm",
                "C#m",
                "Dm",
                "D#m",
                "Em",
                "Fm",
                "F#m",
                "Gm",
                "G#m",
                "Am",
                "A#m",
                "Bm",
            ],
            "schema": {"inputs": [{"name": "key_input"}], "outputs": [{"name": "key_output"}]},
            "inference": {"sample_rate": 44100, "algorithm": "TensorflowPredict2D"},
        }

        # Write metadata files
        discogs_path = metadata_dir / "discogs_metadata.json"
        with open(discogs_path, "w") as f:
            json.dump(discogs_metadata, f)

        genre_path = metadata_dir / "genre_metadata.json"
        with open(genre_path, "w") as f:
            json.dump(genre_metadata, f)

        key_path = metadata_dir / "key_metadata.json"
        with open(key_path, "w") as f:
            json.dump(key_metadata, f)

        return {"discogs": str(discogs_path), "genre": str(genre_path), "key": str(key_path)}

    @pytest.fixture
    def mock_config(self, tmp_path: Path, create_metadata_files: Dict[str, str]) -> Dict[str, Any]:
        """Create a mock plugin configuration."""
        metadata_paths = create_metadata_files

        # Create model directory
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create empty model files
        (models_dir / "discogs_model.pb").touch()
        (models_dir / "genre_model.pb").touch()
        (models_dir / "key_model.pb").touch()
        (models_dir / "tempo_model.pb").touch()

        return {
            "models_directory": str(tmp_path),
            "models": {
                "embeddings": {
                    "discogs": {"metadata_path": metadata_paths["discogs"]},
                    "musicnn": {"model_path": str(models_dir / "musicnn_model.pb"), "model_output": "musicnn_output"},
                },
                "classification": {
                    "genre": {"metadata_path": metadata_paths["genre"]},
                    "mood": {
                        "model_path": str(models_dir / "mood_model.pb"),
                        "model_input": "mood_input",
                        "model_output": "mood_output",
                        "embedding_model": "discogs",
                        "classes": ["happy", "sad", "neutral"],
                        "sample_rate": 22050,
                    },
                },
                "rhythm": {"tempocnn": {"model_path": str(models_dir / "tempo_model.pb")}},
                "harmony": {"key": {"metadata_path": metadata_paths["key"]}},
            },
            "storage": {"database": {"store_probabilities": True, "beat_resolution": 0.001, "chord_format": "simple"}},
        }

    @pytest.fixture
    def mock_model_manager(self, mock_config: Dict[str, Any]) -> MagicMock:
        """Create a mock model manager with metadata support."""
        manager = MagicMock(spec=ModelManager)

        # Configure the has_model method to return True for specific models
        def has_model_side_effect(category: str, model_name: str) -> bool:
            if category == "classification" and model_name in ["genre", "mood"]:
                return True
            if category == "rhythm" and model_name in ["tempocnn"]:
                return True
            if category == "harmony" and model_name in ["key"]:
                return True
            return False

        manager.has_model.side_effect = has_model_side_effect

        # Configure get_model to return mock models
        def get_model_side_effect(category: str, model_name: str) -> Optional[MagicMock]:
            model = MagicMock(spec=TensorflowPredict2D)

            # Configure model behavior based on type
            if category == "classification":
                if model_name == "genre":
                    model.return_value = np.array([0.1, 0.7, 0.1, 0.1])
                elif model_name == "mood":
                    model.return_value = np.array([0.3, 0.6, 0.1])
            elif category == "rhythm" and model_name == "tempocnn":
                model.return_value = np.array([120.5])
            elif category == "harmony" and model_name == "key":
                model.return_value = np.array([0.05] * 12 + [0.1, 0.8, 0.05] * 4)

            return model

        manager.get_model.side_effect = get_model_side_effect

        # Configure get_embedding_model to return mock embedding models
        def get_embedding_model_side_effect(category: str, model_name: str) -> Optional[MagicMock]:
            if category == "classification":
                if model_name == "genre" or model_name == "mood":
                    model = MagicMock(spec=TensorflowPredictEffnetDiscogs)
                    model.return_value = np.random.random(1280)  # Typical embedding size
                    return model
            return None

        manager.get_embedding_model.side_effect = get_embedding_model_side_effect

        # Configure get_model_config to return model configurations
        def get_model_config_side_effect(category: str, model_name: str) -> Optional[ModelConfiguration]:
            if category == "classification" and model_name == "genre":
                config = MagicMock(spec=ModelConfiguration)
                config.sample_rate = 16000
                return config
            elif category == "classification" and model_name == "mood":
                config = MagicMock(spec=ModelConfiguration)
                config.sample_rate = 22050
                return config
            elif category == "harmony" and model_name == "key":
                config = MagicMock(spec=ModelConfiguration)
                config.sample_rate = 44100
                return config
            elif category == "rhythm" and model_name == "tempocnn":
                config = MagicMock(spec=ModelConfiguration)
                config.sample_rate = 44100
                return config
            return None

        manager.get_model_config.side_effect = get_model_config_side_effect

        # Configure get_sample_rate to return sample rates from metadata
        def get_sample_rate_side_effect(category: str, model_name: str) -> int:
            if category == "classification" and model_name == "genre":
                return 16000
            elif category == "classification" and model_name == "mood":
                return 22050
            elif category == "harmony" and model_name == "key":
                return 44100
            else:
                return 44100  # Default

        manager.get_sample_rate.side_effect = get_sample_rate_side_effect

        # Configure get_class_names to return class names from metadata
        def get_class_names_side_effect(category: str, model_name: str) -> List[str]:
            if category == "classification" and model_name == "genre":
                return ["rock", "pop", "jazz", "classical"]
            elif category == "classification" and model_name == "mood":
                return ["happy", "sad", "neutral"]
            elif category == "harmony" and model_name == "key":
                return [
                    "C",
                    "C#",
                    "D",
                    "D#",
                    "E",
                    "F",
                    "F#",
                    "G",
                    "G#",
                    "A",
                    "A#",
                    "B",
                    "Cm",
                    "C#m",
                    "Dm",
                    "D#m",
                    "Em",
                    "Fm",
                    "F#m",
                    "Gm",
                    "G#m",
                    "Am",
                    "A#m",
                    "Bm",
                ]
            return []

        manager.get_class_names.side_effect = get_class_names_side_effect

        return manager

    @pytest.fixture
    def audio_processor(self, mock_model_manager: MagicMock, mock_config: Dict[str, Any]) -> AudioProcessor:
        """Create an AudioProcessor instance with mock dependencies."""
        return AudioProcessor(mock_model_manager, mock_config)

    def test_get_available_features(self, audio_processor: AudioProcessor) -> None:
        """Test determining available features."""
        available = audio_processor._available_features

        # Based on our mock model manager setup
        assert "genre" in available
        assert "mood" in available
        assert "bpm" in available
        assert "key" in available
        assert "style" not in available  # We didn't configure the model manager to have this
        assert "chords" not in available  # Model exists but get_model returns None

    def test_process_file_nonexistent(self, audio_processor: AudioProcessor, tmp_path: Path) -> None:
        """Test processing a nonexistent file."""
        nonexistent_path = tmp_path / "nonexistent.mp3"

        with pytest.raises(AudioProcessingError, match="File does not exist"):
            audio_processor.process_file(str(nonexistent_path))

    @patch("beetsplug.essentia_tensorflow.audio_processor.MonoLoader")
    def test_process_file(
        self, mock_loader: MagicMock, audio_processor: AudioProcessor, mock_audio_data: np.ndarray, tmp_path: Path
    ) -> None:
        """Test processing a valid file."""
        # Create a dummy audio file
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # Configure the mock loader to return our test audio data
        loader_instance = MagicMock()
        loader_instance.return_value = mock_audio_data
        mock_loader.return_value = loader_instance

        # Process the file
        results = audio_processor.process_file(str(test_file))

        # Check that we have results for all expected features
        assert "genre" in results
        assert "mood" in results
        assert "bpm" in results
        assert "key" in results

        # Verify the contents of each result
        assert results["genre"]["feature_type"] == "genre"
        assert "value" in results["genre"]
        assert "confidence" in results["genre"]

        # Check that the correct class labels were used from metadata
        assert results["genre"]["value"] == "pop"  # From the mock model's output

        assert results["mood"]["feature_type"] == "mood"
        assert "value" in results["mood"]

        assert results["bpm"]["feature_type"] == "bpm"
        assert isinstance(results["bpm"]["value"], float)

        assert results["key"]["feature_type"] == "key"
        assert "value" in results["key"]
        assert "confidence" in results["key"]

        # Check that key uses proper labels from metadata
        mock_loader.assert_any_call(
            filename=str(test_file),
            sampleRate=16000,  # Should use the sample rate from metadata for genre model
            resampleQuality=4,
        )

    @patch("beetsplug.essentia_tensorflow.audio_processor.MonoLoader")
    def test_audio_loading_error(self, mock_loader: MagicMock, audio_processor: AudioProcessor, tmp_path: Path) -> None:
        """Test handling of audio loading errors."""
        # Create a dummy audio file
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # Configure the loader to raise an exception for all calls
        mock_loader.side_effect = Exception("Test audio loading error")

        # Process the file - should raise AudioProcessingError
        with pytest.raises(AudioProcessingError, match="No features could be extracted"):
            audio_processor.process_file(str(test_file))

    @patch("beetsplug.essentia_tensorflow.audio_processor.MonoLoader")
    def test_extract_classification_feature(
        self, mock_loader: MagicMock, audio_processor: AudioProcessor, mock_audio_data: np.ndarray
    ) -> None:
        """Test classification feature extraction with metadata labels."""
        with (
            patch.object(audio_processor._model_manager, "get_model") as mock_get_model,
            patch.object(audio_processor._model_manager, "get_embedding_model") as mock_get_embedding,
            patch.object(audio_processor._model_manager, "get_class_names") as mock_get_class_names,
        ):
            # Configure mock models
            model = MagicMock()
            model.return_value = np.array([0.1, 0.7, 0.1, 0.1])
            mock_get_model.return_value = model

            # Configure class names from metadata
            mock_get_class_names.return_value = ["rock", "pop", "jazz", "classical"]

            # Test with embedding model
            embedding_model = MagicMock(spec=TensorflowPredictEffnetDiscogs)
            embedding_model.return_value = np.random.random(1280)
            mock_get_embedding.return_value = embedding_model

            result = audio_processor._extract_classification_feature(
                mock_audio_data,
                "classification",
                "genre",
                "genre",
                {"sample_rate": 16000, "segment_duration": 10, "resample_quality": 4},
            )

            assert result is not None
            assert result.feature_type == "genre"
            assert result.model_name == "genre"
            assert result.labels == ["rock", "pop", "jazz", "classical"]
            assert len(result.values) == 4
            assert result.probabilities == [0.1, 0.7, 0.1, 0.1]

            # Test with no class names in metadata (should generate placeholder labels)
            mock_get_class_names.return_value = []

            result = audio_processor._extract_classification_feature(
                mock_audio_data,
                "classification",
                "genre",
                "genre",
                {"sample_rate": 16000, "segment_duration": 10, "resample_quality": 4},
            )

            assert result is not None
            assert result.labels == ["genre_0", "genre_1", "genre_2", "genre_3"]

    @patch("beetsplug.essentia_tensorflow.audio_processor.MonoLoader")
    def test_extract_key(
        self, mock_loader: MagicMock, audio_processor: AudioProcessor, mock_audio_data: np.ndarray
    ) -> None:
        """Test key extraction with metadata labels."""
        with (
            patch.object(audio_processor._model_manager, "get_model") as mock_get_model,
            patch.object(audio_processor._model_manager, "get_class_names") as mock_get_class_names,
        ):
            # Configure mock model
            model = MagicMock()
            model.return_value = np.array([0.05] * 12 + [0.1, 0.8, 0.05] * 4)
            mock_get_model.return_value = model

            # Configure class names from metadata
            key_labels = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
                "Cm",
                "C#m",
                "Dm",
                "D#m",
                "Em",
                "Fm",
                "F#m",
                "Gm",
                "G#m",
                "Am",
                "A#m",
                "Bm",
            ]
            mock_get_class_names.return_value = key_labels

            result = audio_processor._extract_key(
                mock_audio_data, "harmony", "key", {"sample_rate": 44100, "segment_duration": 30, "resample_quality": 4}
            )

            assert result is not None
            assert result.feature_type == "key"
            assert result.model_name == "key"
            assert result.labels == key_labels

            # Test with no class names in metadata (should use default key labels)
            mock_get_class_names.return_value = []

            result = audio_processor._extract_key(
                mock_audio_data, "harmony", "key", {"sample_rate": 44100, "segment_duration": 30, "resample_quality": 4}
            )

            assert result is not None
            assert len(result.labels) == 24
            assert "C" in result.labels
            assert "Cm" in result.labels

    def test_extract_feature_sample_rate(self, audio_processor: AudioProcessor, tmp_path: Path) -> None:
        """Test that appropriate sample rates from metadata are used."""
        # Create a test file
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # Patch the loading function to check sample rates
        with (
            patch.object(audio_processor, "_load_audio") as mock_load_audio,
            patch.object(audio_processor, "_extract_classification_feature") as mock_extract_class,
        ):
            mock_load_audio.return_value = np.random.random(44100 * 10)
            mock_extract_class.return_value = None

            # Test extraction for genre which should use 16000 Hz from metadata
            audio_processor._extract_feature(str(test_file), "genre")
            mock_load_audio.assert_called_with(
                str(test_file),
                16000,  # Sample rate from metadata
                4,  # Default resample quality
            )

            # Test extraction for mood which should use 22050 Hz from config
            audio_processor._extract_feature(str(test_file), "mood")
            mock_load_audio.assert_called_with(
                str(test_file),
                22050,  # Sample rate from direct config
                4,  # Default resample quality
            )
