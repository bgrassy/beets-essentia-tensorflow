"""Tests for model management functionality."""

from typing import Any
from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from collections.abc import Generator

import pytest
from beets import ui

from beetsplug.essentia_tensorflow.model_manager import (
    ModelConfiguration,
    ModelManager,
    ModelLoadError,
)
from beets.test.helper import capture_log


class TestModelConfiguration:
    """Tests for the ModelConfiguration class."""

    def test_initialization(self) -> None:
        """Test initialization with different configurations."""
        # Test with complete configuration

        config = ModelConfiguration(
            {"model_path": "/path/to/model", "embedding": "discogs", "model_output": "custom_output"},
        )
        assert config.model_path == "/path/to/model"
        assert config.embedding == "discogs"
        assert config.model_output == "custom_output"

        # Test with minimal configuration
        config = ModelConfiguration({"model_path": "/path/to/model"})
        assert config.model_path == "/path/to/model"
        assert config.embedding is None
        assert config.model_output == "model/Softmax"  # Default

        # Test with empty configuration
        config = ModelConfiguration({})
        assert config.model_path is None
        assert config.embedding is None
        assert config.model_output == "model/Softmax"

    def test_validation_methods(self) -> None:
        """Test validation methods."""
        # Valid configuration with embedding reference
        config = ModelConfiguration({"model_path": "/path/to/model", "embedding": "discogs"})
        assert config.is_valid() is True
        assert config.has_embedding_reference() is True

        # Valid configuration without embedding reference
        config = ModelConfiguration({"model_path": "/path/to/model"})
        assert config.is_valid() is True
        assert config.has_embedding_reference() is False

        # Invalid configuration
        config = ModelConfiguration({})
        assert config.is_valid() is False
        assert config.has_embedding_reference() is False


class TestModelManager:
    """Tests for the ModelManager class."""

    @pytest.fixture
    def tmp_path(self) -> Generator[Path, None, None]:
        """Create a temporary directory for test models."""
        tmp_dir = tempfile.mkdtemp()
        yield Path(tmp_dir)
        shutil.rmtree(tmp_dir)

    @pytest.fixture
    def mock_models(self) -> Generator[dict[str, MagicMock], None, None]:
        """Mock of the Essentia model classes."""
        with (
            patch("beetsplug.essentia_tensorflow.model_manager.TensorflowPredictEffnetDiscogs") as discogs_mock,
            patch("beetsplug.essentia_tensorflow.model_manager.TensorflowPredict2D") as standard_mock,
        ):
            yield {"discogs": discogs_mock, "standard": standard_mock}

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> dict[str, Any]:
        """Create a mock plugin configuration with temporary paths."""
        # Create test directories and files
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Create test model files
        genre_model_path = model_dir / "genre_model"
        mood_model_path = model_dir / "mood_model"
        discogs_model_path = model_dir / "discogs_model"
        musicnn_model_path = model_dir / "musicnn_model"

        # Create empty files to pass path validation
        genre_model_path.touch()
        mood_model_path.touch()
        discogs_model_path.touch()
        musicnn_model_path.touch()

        return {
            "models": {
                "embeddings": {
                    "discogs": {"model_path": str(discogs_model_path), "model_output": "PartitionedCall:1"},
                    "musicnn": {"model_path": str(musicnn_model_path)},
                },
                "classification": {
                    "genre": {
                        "model_path": str(genre_model_path),
                        "embedding": "discogs",
                        "model_output": "genre_output",
                    },
                    "mood": {"model_path": str(mood_model_path), "embedding": "musicnn"},
                    "style": {},  # Empty config
                },
            },
        }

    @pytest.fixture
    def model_manager(self, mock_config: MagicMock, mock_models: MagicMock) -> ModelManager:
        """Create a ModelManager instance with mock config."""
        # We need to patch the initialization to prevent actual model loading
        with patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._preload_models"):
            manager = ModelManager(mock_config)
            # Manually set the models that would have been preloaded
            manager._models = {"classification.genre": MagicMock(), "classification.mood": MagicMock()}
            manager._embeddings = {"embeddings.discogs": MagicMock(), "embeddings.musicnn": MagicMock()}
            return manager

    def test_validate_model_paths(self, model_manager: ModelManager, tmp_path: Path) -> None:
        """Test model path validation."""
        # Valid paths should not raise exceptions
        model_manager._validate_model_paths()

        # Test invalid model path
        model_manager._config["models"]["classification"]["genre"]["model_path"] = "/nonexistent/path"
        with pytest.raises(ui.UserError, match="Model path not found"):
            model_manager._validate_model_paths()

        # Reset model path and test invalid embedding model
        genre_model_path = tmp_path / "models" / "genre_model"
        model_manager._config["models"]["classification"]["genre"]["model_path"] = str(genre_model_path)
        model_manager._config["models"]["classification"]["genre"]["embedding"] = "nonexistent"
        with pytest.raises(ui.UserError, match="Referenced embedding model 'nonexistent' not found"):
            model_manager._validate_model_paths()

        # Test invalid embedding model path
        model_manager._config["models"]["classification"]["genre"]["embedding"] = "discogs"
        model_manager._config["models"]["embeddings"]["discogs"]["model_path"] = "/nonexistent/path"
        with pytest.raises(ui.UserError, match="Embedding model path not found"):
            model_manager._validate_model_paths()

    def test_preload_models(self, mock_config: MagicMock, mock_models: MagicMock) -> None:
        """Test that models are preloaded during initialization."""
        with patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._validate_model_paths"):
            # Create manager without mocking _preload_models
            manager = ModelManager(mock_config)

            # Check that the expected models were loaded
            assert "classification.genre" in manager._models
            assert "classification.mood" in manager._models
            assert "classification.style" not in manager._models

            # Check that embedding models were loaded
            assert "embeddings.discogs" in manager._embeddings
            assert "embeddings.musicnn" in manager._embeddings

    def test_load_discogs_embedding_model(self, model_manager: ModelManager, mock_models: MagicMock) -> None:
        """Test loading a Discogs embedding model."""
        # Set up mock model
        discogs_mock = mock_models["discogs"].return_value

        # Test successful load with default output
        model = model_manager._load_discogs_embedding_model_from_path("/path/to/model")
        assert model is discogs_mock

        # Verify correct parameters
        mock_models["discogs"].assert_called_with(graphFilename="/path/to/model", output="PartitionedCall:1")

        # Test with custom output
        model = model_manager._load_discogs_embedding_model_from_path("/path/to/model", "custom_output")
        mock_models["discogs"].assert_called_with(graphFilename="/path/to/model", output="custom_output")

        # Test error handling
        mock_models["discogs"].side_effect = RuntimeError("Test error")

        with pytest.raises(ModelLoadError):
            model_manager._load_discogs_embedding_model_from_path("/path/to/model")

    def test_load_model(self, model_manager: ModelManager, mock_models: MagicMock) -> None:
        """Test loading a standard model."""
        # Set up mock model
        standard_mock = mock_models["standard"].return_value

        # Test successful load with default output
        model = model_manager._load_model_from_path("/path/to/model")
        assert model is standard_mock

        # Verify correct parameters
        mock_models["standard"].assert_called_with(graphFilename="/path/to/model", output="model/Softmax")

        # Test with custom output
        model = model_manager._load_model_from_path("/path/to/model", "custom_output")
        mock_models["standard"].assert_called_with(graphFilename="/path/to/model", output="custom_output")

        # Test error handling
        mock_models["standard"].side_effect = RuntimeError("Test error")

        with pytest.raises(ModelLoadError):
            model_manager._load_model_from_path("/path/to/model")

    def test_get_model(self, model_manager: ModelManager) -> None:
        """Test getting a loaded model."""
        # Test getting existing models
        assert model_manager.get_model("classification", "genre") is not None
        assert model_manager.get_model("classification", "mood") is not None

        # Test getting non-existent models
        assert model_manager.get_model("classification", "style") is None

    def test_get_embedding_model(self, model_manager: ModelManager) -> None:
        """Test getting an associated embedding model."""
        # Genre uses discogs embedding
        embedding_model = model_manager.get_embedding_model("classification", "genre")
        assert embedding_model is model_manager._embeddings["embeddings.discogs"]

        # Mood uses musicnn embedding
        embedding_model = model_manager.get_embedding_model("classification", "mood")
        assert embedding_model is model_manager._embeddings["embeddings.musicnn"]

        # Style has no configuration, so no embedding
        assert model_manager.get_embedding_model("classification", "style") is None

        # Test with config but no embedding reference
        model_manager._config["models"]["classification"]["nonexistent"] = {"model_path": "/some/path"}
        assert model_manager.get_embedding_model("classification", "nonexistent") is None

    def test_get_embedding_model_by_name(self, model_manager: ModelManager) -> None:
        """Test getting an embedding model directly by name."""
        # Test existing embedding models
        assert model_manager.get_embedding_model_by_name("discogs") is not None
        assert model_manager.get_embedding_model_by_name("musicnn") is not None

        # Test non-existent embedding model
        assert model_manager.get_embedding_model_by_name("nonexistent") is None

    def test_get_model_config(self, model_manager: ModelManager, mock_config: MagicMock) -> None:
        """Test getting model configuration."""
        # Test getting existing configuration
        genre_config = model_manager.get_model_config("classification", "genre")
        assert isinstance(genre_config, ModelConfiguration)
        assert genre_config.model_path == mock_config["models"]["classification"]["genre"]["model_path"]
        assert genre_config.embedding == "discogs"
        assert genre_config.model_output == "genre_output"

        # Test getting configuration without embedding
        model_manager._config["models"]["classification"]["other"] = {"model_path": "/some/path"}
        other_config = model_manager.get_model_config("classification", "other")
        assert isinstance(other_config, ModelConfiguration)
        assert other_config.model_path == "/some/path"
        assert other_config.embedding is None
        assert other_config.model_output == "model/Softmax"  # Default

        # Test getting non-existent configuration
        assert model_manager.get_model_config("classification", "nonexistent") is None

    def test_has_model(self, model_manager: ModelManager) -> None:
        """Test checking if a model is loaded."""
        # Test existing models
        assert model_manager.has_model("classification", "genre") is True
        assert model_manager.has_model("classification", "mood") is True

        # Test non-existent models
        assert model_manager.has_model("classification", "style") is False

    def test_has_embedding_model(self, model_manager: ModelManager) -> None:
        """Test checking if a model has an associated embedding model."""
        # Test models with embedding references
        assert model_manager.has_embedding_model("classification", "genre") is True
        assert model_manager.has_embedding_model("classification", "mood") is True

        # Test model without embedding reference
        assert model_manager.has_embedding_model("classification", "style") is False

        # Test with a valid config but no embedding
        model_manager._config["models"]["classification"]["other"] = {"model_path": "/some/path"}
        assert model_manager.has_embedding_model("classification", "other") is False

        # Test with embedding reference to non-existent embedding
        model_manager._config["models"]["classification"]["invalid"] = {
            "model_path": "/some/path",
            "embedding": "nonexistent",
        }
        assert model_manager.has_embedding_model("classification", "invalid") is False

    def test_error_handling_during_preload(self, mock_config: MagicMock, mock_models: MagicMock) -> None:
        """Test error handling during model preloading."""
        # Make model loading fail
        mock_models["standard"].side_effect = RuntimeError("Test error")

        # Attempt to create manager, which should fail during preloading
        with (
            pytest.raises(ui.UserError, match="Failed to preload"),
            capture_log() as logs,
            patch("beetsplug.essentia_tensorflow.model_manager.ModelManager._validate_model_paths"),
        ):
            ModelManager(mock_config)

        # Check for error log
        assert "Failed to preload" in "".join(logs)
