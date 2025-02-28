"""Tests for updated model management functionality."""

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from beets import ui

from beetsplug.essentia_tensorflow.model_manager import (
    ModelConfiguration,
    ModelManager,
    ModelLoadError,
    ModelMetadataError,
)


class TestModelConfiguration:
    """Tests for the updated ModelConfiguration class."""

    @pytest.fixture
    def tmp_path(self):
        """Create a temporary directory for test files."""
        tmp_dir = tempfile.mkdtemp()
        yield Path(tmp_dir)
        shutil.rmtree(tmp_dir)

    @pytest.fixture
    def metadata_json(self, tmp_path):
        """Create a sample metadata JSON file."""
        metadata = {
            "name": "Test Model",
            "type": "Music genre classification",
            "model_path": str(tmp_path / "model_file.pb"),
            "version": "1",
            "classes": ["rock", "pop", "jazz", "classical"],
            "schema": {
                "inputs": [{"name": "test_input", "type": "float", "shape": ["batch_size", 1280]}],
                "outputs": [{"name": "test_output", "type": "float", "shape": ["batch_size", 4], "op": "Sigmoid"}],
            },
            "inference": {
                "sample_rate": 16000,
                "algorithm": "TensorflowPredict2D",
                "embedding_model": {"algorithm": "TensorflowPredictEffnetDiscogs", "model_name": "discogs"},
            },
        }

        metadata_path = tmp_path / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create an empty model file
        (tmp_path / "model_file.pb").touch()

        return metadata_path

    def test_init_from_direct_config(self):
        """Test initialization with direct configuration."""
        config = {
            "model_path": "/path/to/model",
            "embedding_model": "discogs",
            "model_output": "custom_output",
            "model_input": "custom_input",
            "sample_rate": 48000,
        }

        model_config = ModelConfiguration(config)

        assert model_config.model_path == "/path/to/model"
        assert model_config.embedding == "discogs"
        assert model_config.model_output == "custom_output"
        assert model_config.model_input == "custom_input"
        assert model_config.sample_rate == 48000
        assert model_config.metadata is None
        assert model_config.classes is None

    def test_init_from_metadata(self, metadata_json):
        """Test initialization from metadata JSON."""
        config = {"metadata_path": str(metadata_json)}

        model_config = ModelConfiguration(config)

        assert model_config.metadata is not None
        assert model_config.model_path is not None
        assert model_config.metadata["name"] == "Test Model"
        assert model_config.model_input == "test_input"
        assert model_config.model_output == "test_output"
        assert model_config.sample_rate == 16000
        assert model_config.embedding == "discogs"
        assert model_config.classes == ["rock", "pop", "jazz", "classical"]

    def test_override_from_config(self, metadata_json):
        """Test overriding metadata with direct configuration."""
        config = {
            "metadata_path": str(metadata_json),
            "model_path": "/override/path",
            "model_output": "override_output",
            "sample_rate": 22050,
        }

        model_config = ModelConfiguration(config)

        # Check that metadata was loaded
        assert model_config.metadata is not None

        # Check that direct config overrides metadata
        assert model_config.model_path == "/override/path"
        assert model_config.model_output == "override_output"
        assert model_config.sample_rate == 22050

        # Check that non-overridden values are from metadata
        assert model_config.model_input == "test_input"
        assert model_config.embedding == "discogs"

    def test_relative_paths_with_base_dir(self, tmp_path, metadata_json):
        """Test resolving relative paths with base directory."""
        # Create base dir and subdirectories for models
        base_dir = tmp_path / "models"
        base_dir.mkdir()

        # Create a model file in the base dir
        model_file = base_dir / "test_model.pb"
        model_file.touch()

        # Create relative path config
        config = {"model_path": "test_model.pb"}

        # Initialize with base directory
        model_config = ModelConfiguration(config, str(base_dir))

        # Check path was properly resolved
        assert model_config.model_path == str(model_file)

    def test_invalid_metadata_file(self, tmp_path):
        """Test handling of invalid metadata file."""
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("invalid json content")

        config = {"metadata_path": str(invalid_path)}

        with pytest.raises(ModelMetadataError):
            ModelConfiguration(config)

    def test_validation_methods(self, metadata_json):
        """Test validation methods."""
        # Valid with model path from metadata
        config = {"metadata_path": str(metadata_json)}
        model_config = ModelConfiguration(config)
        assert model_config.is_valid() is True
        assert model_config.has_embedding_reference() is True

        # Valid with direct model path
        config = {"model_path": "/path/to/model"}
        model_config = ModelConfiguration(config)
        assert model_config.is_valid() is True
        assert model_config.has_embedding_reference() is False

        # Invalid without model path
        config = {}
        model_config = ModelConfiguration(config)
        assert model_config.is_valid() is False

        # Valid model path with direct embedding reference
        config = {"model_path": "/path/to/model", "embedding_model": "discogs"}
        model_config = ModelConfiguration(config)
        assert model_config.has_embedding_reference() is True

    def test_get_class_names(self, metadata_json):
        """Test getting class names."""
        # With classes from metadata
        config = {"metadata_path": str(metadata_json)}
        model_config = ModelConfiguration(config)
        assert model_config.get_class_names() == ["rock", "pop", "jazz", "classical"]

        # With direct classes
        config = {"model_path": "/path/to/model", "classes": ["a", "b", "c"]}
        model_config = ModelConfiguration(config)
        assert model_config.get_class_names() == ["a", "b", "c"]

        # Without classes
        config = {"model_path": "/path/to/model"}
        model_config = ModelConfiguration(config)
        assert model_config.get_class_names() == []


class TestModelManager:
    """Tests for the updated ModelManager class."""

    @pytest.fixture
    def tmp_path(self):
        """Create a temporary directory for test files."""
        tmp_dir = tempfile.mkdtemp()
        yield Path(tmp_dir)
        shutil.rmtree(tmp_dir)

    @pytest.fixture
    def create_model_files(self, tmp_path):
        """Create sample model files and metadata."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Create model files
        (model_dir / "discogs_model.pb").touch()
        (model_dir / "genre_model.pb").touch()
        (model_dir / "mood_model.pb").touch()

        # Create metadata files
        discogs_metadata = {
            "name": "Discogs Embedding",
            "model_path": str(model_dir / "discogs_model.pb"),
            "schema": {"outputs": [{"name": "PartitionedCall:1"}]},
        }

        genre_metadata = {
            "name": "Genre Classifier",
            "model_path": str(model_dir / "genre_model.pb"),
            "classes": ["rock", "pop", "jazz"],
            "schema": {
                "inputs": [{"name": "serving_default_model_Placeholder"}],
                "outputs": [{"name": "PartitionedCall:0"}],
            },
            "inference": {"sample_rate": 16000, "embedding_model": {"model_name": "discogs"}},
        }

        with open(model_dir / "discogs_metadata.json", "w") as f:
            json.dump(discogs_metadata, f)

        with open(model_dir / "genre_metadata.json", "w") as f:
            json.dump(genre_metadata, f)

        return model_dir

    @pytest.fixture
    def mock_config(self, create_model_files):
        """Create a mock plugin configuration."""
        model_dir = create_model_files

        return {
            "models_directory": str(model_dir),
            "models": {
                "embeddings": {
                    "discogs": {"metadata_path": "discogs_metadata.json"},
                    "musicnn": {"model_path": str(model_dir / "musicnn_model.pb"), "model_output": "custom_output"},
                },
                "classification": {
                    "genre": {"metadata_path": "genre_metadata.json"},
                    "style": {"model_path": str(model_dir / "style_model.pb"), "embedding_model": "discogs"},
                    "mood": {
                        "model_path": str(model_dir / "mood_model.pb"),
                        "model_input": "custom_input",
                        "model_output": "custom_output",
                        "classes": ["happy", "sad", "neutral"],
                    },
                    "empty": {},
                },
            },
        }

    @pytest.fixture
    def mock_essentia_models(self):
        """Mock of the Essentia model classes."""
        with (
            patch("beetsplug.essentia_tensorflow.model_manager.TensorflowPredictEffnetDiscogs") as discogs_mock,
            patch("beetsplug.essentia_tensorflow.model_manager.TensorflowPredict2D") as standard_mock,
        ):
            yield {"discogs": discogs_mock, "standard": standard_mock}

    def test_validate_model_paths(self, mock_config, mock_essentia_models, create_model_files):
        """Test model path validation with both metadata and direct configuration."""
        # Touch files that don't exist so validation passes
        model_dir = create_model_files
        (model_dir / "musicnn_model.pb").touch()
        (model_dir / "style_model.pb").touch()

        # Patch _validate_model_paths and _validate_and_get_embedding_models to prevent errors
        with (
            patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._preload_models"),
            patch(
                "beetsplug.essentia_tensorflow.model_manager.ModelManager._validate_and_get_embedding_models"
            ) as mock_validate,
        ):
            # Return some dummy available embeddings
            mock_validate.return_value = {"discogs", "musicnn"}

            manager = ModelManager(mock_config)

            # Test validation with mocked embedding validation
            manager._validate_model_paths()

            # Test non-existent model path in direct config
            mock_config["models"]["classification"]["style"]["model_path"] = str(model_dir / "nonexistent.pb")
            with pytest.raises(ui.UserError, match="not found"):
                manager._validate_model_paths()

    def test_model_config_from_metadata(self, mock_config, mock_essentia_models, create_model_files):
        """Test loading model configuration from metadata."""
        # Touch necessary files
        model_dir = create_model_files
        (model_dir / "musicnn_model.pb").touch()
        (model_dir / "style_model.pb").touch()

        # We need to patch model loading and validation
        with (
            patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._preload_models"),
            patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._validate_model_paths"),
        ):
            manager = ModelManager(mock_config)

            # Create and inject a genre model config
            genre_config = ModelConfiguration(
                {"metadata_path": str(model_dir / "genre_metadata.json")}, str(model_dir.parent)
            )
            manager._model_configs["classification.genre"] = genre_config

            # Get the genre model config that uses metadata
            result_config = manager.get_model_config("classification", "genre")

            # Verify it loaded the metadata correctly
            assert result_config is not None
            assert result_config.metadata is not None
            assert result_config.model_path is not None
            assert result_config.embedding == "discogs"

    def test_model_config_direct(self, mock_config, mock_essentia_models, create_model_files):
        """Test loading model configuration from direct config."""
        # Touch necessary files
        model_dir = create_model_files
        (model_dir / "musicnn_model.pb").touch()
        (model_dir / "style_model.pb").touch()
        (model_dir / "mood_model.pb").touch()

        # We need to patch model loading and validation
        with (
            patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._preload_models"),
            patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._validate_model_paths"),
        ):
            manager = ModelManager(mock_config)

            # Create and inject a mood model config
            mood_config = ModelConfiguration(
                {
                    "model_path": str(model_dir / "mood_model.pb"),
                    "model_output": "custom_output",
                    "model_input": "custom_input",
                    "classes": ["happy", "sad", "neutral"],
                }
            )
            manager._model_configs["classification.mood"] = mood_config

            # Get the mood model config that uses direct config
            result_config = manager.get_model_config("classification", "mood")

            # Verify it loaded the direct config correctly
            assert result_config is not None
            assert result_config.metadata is None
            assert result_config.model_path is not None
            assert result_config.model_input == "custom_input"
            assert result_config.model_output == "custom_output"
            assert result_config.classes == ["happy", "sad", "neutral"]

    def test_preload_models(self, mock_config, mock_essentia_models, create_model_files):
        """Test that models are preloaded during initialization."""
        # Touch required files that don't exist
        model_dir = create_model_files
        (model_dir / "musicnn_model.pb").touch()
        (model_dir / "style_model.pb").touch()
        (model_dir / "mood_model.pb").touch()

        # Configure mock models
        discogs_mock = mock_essentia_models["discogs"]
        standard_mock = mock_essentia_models["standard"]

        # Initialize manager with patched validation
        with patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._validate_model_paths"):
            # Create manually initialized configuration objects
            manager = ModelManager(mock_config)

            # We need to set up model configs before calling _preload_models
            discogs_config = ModelConfiguration(
                {"model_path": str(model_dir / "discogs_model.pb"), "model_output": "PartitionedCall:1"}
            )
            genre_config = ModelConfiguration(
                {
                    "model_path": str(model_dir / "genre_model.pb"),
                    "embedding": "discogs",
                    "model_output": "PartitionedCall:0",
                }
            )
            mood_config = ModelConfiguration(
                {"model_path": str(model_dir / "mood_model.pb"), "model_output": "custom_output"}
            )
            style_config = ModelConfiguration({"model_path": str(model_dir / "style_model.pb"), "embedding": "discogs"})

            # Add configs to manager
            manager._model_configs = {
                "embeddings.discogs": discogs_config,
                "classification.genre": genre_config,
                "classification.mood": mood_config,
                "classification.style": style_config,
            }

            # Now manually call _preload_models
            manager._preload_models()

            # Verify that models were loaded
            assert discogs_mock.called
            assert standard_mock.called

            # Check for existence of expected models
            assert "embeddings.discogs" in manager._embeddings
            assert "classification.genre" in manager._models
            assert "classification.style" in manager._models
            assert "classification.mood" in manager._models

    def test_get_class_names(self, mock_config, mock_essentia_models, create_model_files):
        """Test getting class names from models."""
        # Touch required files that don't exist
        model_dir = create_model_files
        (model_dir / "musicnn_model.pb").touch()
        (model_dir / "style_model.pb").touch()

        # We need to patch model loading
        with patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._preload_models"):
            manager = ModelManager(mock_config)

            # Get class names from metadata-based model
            genre_classes = manager.get_class_names("classification", "genre")
            assert genre_classes == ["rock", "pop", "jazz"]

            # Get class names from direct config
            mood_classes = manager.get_class_names("classification", "mood")
            assert mood_classes == ["happy", "sad", "neutral"]

            # Get class names from model without classes
            style_classes = manager.get_class_names("classification", "style")
            assert style_classes == []

            # Get class names from non-existent model
            nonexistent_classes = manager.get_class_names("classification", "nonexistent")
            assert nonexistent_classes == []

    def test_get_sample_rate(self, mock_config, mock_essentia_models, create_model_files):
        """Test getting sample rate from models."""
        # Touch required files that don't exist
        model_dir = create_model_files
        (model_dir / "musicnn_model.pb").touch()
        (model_dir / "style_model.pb").touch()

        # We need to patch model loading
        with patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._preload_models"):
            manager = ModelManager(mock_config)

            # Get sample rate from metadata-based model
            genre_rate = manager.get_sample_rate("classification", "genre")
            assert genre_rate == 16000

            # Get sample rate from model without specified rate (should return default)
            style_rate = manager.get_sample_rate("classification", "style")
            assert style_rate == 44100

            # Get sample rate from non-existent model (should return default)
            nonexistent_rate = manager.get_sample_rate("classification", "nonexistent")
            assert nonexistent_rate == 44100

    def test_get_embedding_model(self, mock_config, mock_essentia_models, create_model_files):
        """Test getting embedding models for classification models."""
        # Touch required files that don't exist
        model_dir = create_model_files
        (model_dir / "musicnn_model.pb").touch()
        (model_dir / "style_model.pb").touch()

        # We need to patch model loading
        with patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._preload_models"):
            manager = ModelManager(mock_config)

            # Set up mock embedding models
            manager._embeddings = {"embeddings.discogs": MagicMock(), "embeddings.musicnn": MagicMock()}

            # Get embedding model for a model with embedding reference from metadata
            genre_embedding = manager.get_embedding_model("classification", "genre")
            assert genre_embedding is not None
            assert genre_embedding is manager._embeddings["embeddings.discogs"]

            # Get embedding model for a model with direct embedding reference
            style_embedding = manager.get_embedding_model("classification", "style")
            assert style_embedding is not None
            assert style_embedding is manager._embeddings["embeddings.discogs"]

            # Get embedding model for a model without embedding reference
            mood_embedding = manager.get_embedding_model("classification", "mood")
            assert mood_embedding is None

            # Get embedding model for non-existent model
            nonexistent_embedding = manager.get_embedding_model("classification", "nonexistent")
            assert nonexistent_embedding is None
