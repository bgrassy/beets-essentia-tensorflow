"""Model management implementation for Essentia plugin."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from beets import ui
from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs


class ModelLoadError(Exception):
    """Raised when a model fails to load."""


class ModelMetadataError(Exception):
    """Raised when there's an error with model metadata."""


class ModelConfiguration:
    """Represents a model configuration with paths and output settings."""

    def __init__(self, config: Dict[str, Any], base_directory: Optional[str] = None) -> None:
        """Initialize model configuration.

        Args:
        ----
            config: Model configuration dictionary
            base_directory: Optional base directory for resolving relative paths

        """
        self.metadata = None
        self.metadata_path = config.get("metadata_path")
        self.model_path = None
        self.embedding = None
        self.model_output = None
        self.model_input = None
        self.classes = None
        self.sample_rate = None
        self.base_directory = base_directory

        # Load metadata if specified
        if self.metadata_path:
            self._load_metadata(self.metadata_path)

        # Apply override parameters from config (these take precedence over metadata)
        self._apply_overrides(config)

    def _load_metadata(self, metadata_path: str) -> None:
        """Load model metadata from JSON file.

        Args:
        ----
            metadata_path: Path to the metadata JSON file

        Raises:
        ------
            ModelMetadataError: If metadata cannot be loaded or is invalid

        """
        try:
            # Resolve path against base directory if it's relative
            if self.base_directory and not Path(metadata_path).is_absolute():
                full_path = Path(self.base_directory) / metadata_path
            else:
                full_path = Path(metadata_path)

            with open(full_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            # Extract required fields from metadata
            if "model_path" in self.metadata:
                # Resolve model path against base directory if it's relative
                model_path = self.metadata["model_path"]
                if self.base_directory and not Path(model_path).is_absolute():
                    self.model_path = str(Path(self.base_directory) / model_path)
                else:
                    self.model_path = model_path

            # Extract schema information
            schema = self.metadata.get("schema", {})
            inputs = schema.get("inputs", [{}])[0]
            outputs = schema.get("outputs", [{}])[0]

            self.model_input = inputs.get("name", "model/Placeholder")
            self.model_output = outputs.get("name", "model/Softmax")

            # Extract inference information
            inference = self.metadata.get("inference", {})
            self.sample_rate = inference.get("sample_rate", 44100)

            # Extract embedding model information
            embedding_info = inference.get("embedding_model", {})
            if embedding_info:
                self.embedding = embedding_info.get("model_name")

            # Extract class labels if available
            self.classes = self.metadata.get("classes", [])

        except (json.JSONDecodeError, IOError) as e:
            raise ModelMetadataError(f"Failed to load metadata from {metadata_path}: {str(e)}") from e
        except KeyError as e:
            raise ModelMetadataError(f"Missing required field in metadata: {str(e)}") from e

    def _apply_overrides(self, config: Dict[str, Any]) -> None:
        """Apply override parameters from configuration.

        Args:
        ----
            config: Configuration dictionary with override parameters

        """
        # Apply model path override if specified
        if "model_path" in config:
            model_path = config["model_path"]
            # Resolve path against base directory if it's relative
            if self.base_directory and not Path(model_path).is_absolute():
                self.model_path = str(Path(self.base_directory) / model_path)
            else:
                self.model_path = model_path

        # Apply other overrides if specified
        if "embedding_model" in config:
            self.embedding = config["embedding_model"]
        if "model_output" in config:
            self.model_output = config["model_output"]
        if "model_input" in config:
            self.model_input = config["model_input"]
        if "sample_rate" in config:
            self.sample_rate = config["sample_rate"]
        if "classes" in config:
            self.classes = config["classes"]

    def is_valid(self) -> bool:
        """Check if model configuration has at least a model path."""
        return bool(self.model_path)

    def has_embedding_reference(self) -> bool:
        """Check if model configuration references an embedding model."""
        return bool(self.embedding)

    def get_class_names(self) -> list[str]:
        """Get the class names for classification models.

        Returns:
        -------
            List of class names or empty list if not available

        """
        return self.classes or []


class ModelManager:
    """Manages loading and validation of TensorFlow models."""

    def __init__(self, plugin_config: Dict[str, Any]) -> None:
        """Initialize the model manager and preload all configured models.

        Args:
        ----
            plugin_config: Plugin configuration dictionary

        """
        self._config = plugin_config
        self._log = logging.getLogger("beets.essentia.models")
        self._models: Dict[str, Any] = {}  # Store loaded models
        self._model_configs: Dict[str, ModelConfiguration] = {}  # Store model configurations
        self._embeddings: Dict[str, Any] = {}  # Store loaded embedding models

        # Get base directory for models if specified
        self._models_base_dir = self._config.get("models_directory")

        # Validate and preload all models
        self._validate_model_paths()
        self._preload_models()

    def _validate_model_paths(self) -> None:
        """Validate that all configured model paths exist and embedding references are valid."""
        models_config = self._config["models"]

        # First validate all embedding models as they will be referenced
        if "embeddings" in models_config:
            available_embeddings = self._validate_and_get_embedding_models(models_config["embeddings"])

        # Now validate other models and their embedding references
        for category in models_config:
            if category == "embeddings":
                continue  # Already validated

            self._validate_model_config(category, models_config[category], available_embeddings)

    def _validate_and_get_embedding_models(self, embeddings_config: Dict[str, Any]) -> set[str]:
        available_embeddings = set()
        for model_name, model_config in embeddings_config.items():
            try:
                # Create model configuration with base directory
                config = ModelConfiguration(model_config, self._models_base_dir)

                if not config.is_valid():
                    self._log.warning(f"Skipping embedding model {model_name} due to invalid configuration")
                    continue

                # Validate model path exists
                model_path = Path(config.model_path).expanduser()
                if not model_path.exists():
                    msg = f"Embedding model path not found: {model_path} for embeddings.{model_name}"
                    raise ui.UserError(msg)

                # Store configuration for later reference
                self._model_configs[f"embeddings.{model_name}"] = config

                # Add to available embeddings
                available_embeddings.add(model_name)

            except ModelMetadataError as e:
                msg = f"Error in metadata for embeddings.{model_name}: {str(e)}"
                raise ui.UserError(msg) from e

        return available_embeddings

    def _validate_model_config(
        self,
        category: str,
        category_model_configs: Dict[str, Any],
        available_embeddings: set[str],
    ) -> None:
        """Validate all model configurations of a specific model category."""
        for model_name, model_config in category_model_configs.items():
            try:
                # Skip if empty config
                if not model_config:
                    continue

                # Create model configuration with base directory
                config = ModelConfiguration(model_config, self._models_base_dir)
                model_id = f"{category}.{model_name}"

                # Store configuration for later reference
                self._model_configs[model_id] = config

                # Skip validation if configuration is not valid
                # (will be skipped during model loading)
                if not config.is_valid():
                    self._log.warning(f"Skipping model {model_id} due to invalid configuration")
                    continue

                # Validate model path
                model_path = Path(config.model_path).expanduser()
                if not model_path.exists():
                    msg = f"Model path not found: {model_path} for {category}.{model_name}"
                    raise ui.UserError(msg)

                # Validate embedding reference
                if config.has_embedding_reference():
                    embedding_name = config.embedding
                    if embedding_name not in available_embeddings:
                        msg = f"Referenced embedding model '{embedding_name}' not found for {category}.{model_name}"
                        raise ui.UserError(msg)

            except ModelMetadataError as e:
                msg = f"Error in metadata for {category}.{model_name}: {str(e)}"
                raise ui.UserError(msg) from e

    def _preload_models(self) -> None:
        """Preload all configured models."""
        models_config = self._config["models"]

        # First load all embedding models
        if "embeddings" in models_config:
            for model_name, _ in models_config["embeddings"].items():
                embedding_id = f"embeddings.{model_name}"
                config = self._model_configs.get(embedding_id)

                # Skip if no valid configuration
                if not config or not config.is_valid():
                    self._log.info(f"Skipping embedding model {model_name} due to invalid configuration")
                    continue

                self._load_embedding_model(model_name, config)

        # Now load all other models
        for category in models_config:
            if category == "embeddings":
                continue  # Already loaded

            for model_name, _ in models_config[category].items():
                model_id = f"{category}.{model_name}"
                config = self._model_configs.get(model_id)

                # Skip if no valid configuration
                if not config or not config.is_valid():
                    self._log.info(f"Skipping model {model_id} due to invalid configuration")
                    continue

                self._load_model(config, model_id)

    def _load_embedding_model(self, model_name: str, config: ModelConfiguration) -> None:
        """Load embedding model of given name from config."""
        embedding_id = f"embeddings.{model_name}"
        model_path = str(Path(config.model_path).expanduser())

        try:
            # Special case for discogs embedding model
            if model_name == "discogs":
                model_output = config.model_output or "PartitionedCall:1"
                self._embeddings[embedding_id] = self._load_discogs_embedding_model_from_path(model_path, model_output)
            else:
                # Other embedding types
                model_output = config.model_output or "model/Softmax"
                model_input = config.model_input or "model/Placeholder"
                self._embeddings[embedding_id] = self._load_model_from_path(model_path, model_input, model_output)

            self._log.info(f"Preloaded embedding model: {embedding_id}")
        except Exception as e:
            msg = f"Failed to preload embedding model {embedding_id}"
            self._log.error(msg)
            raise ui.UserError(msg) from e

    def _load_model(self, config: ModelConfiguration, model_id: str) -> None:
        """Load generic Tensorflow model of given name from config."""
        model_path = str(Path(config.model_path).expanduser())
        model_output = config.model_output or "model/Softmax"
        model_input = config.model_input or "model/Placeholder"

        try:
            # Load the model
            self._models[model_id] = self._load_model_from_path(model_path, model_input, model_output)
            self._log.info(f"Preloaded model: {model_id}")
        except Exception as e:
            msg = f"Failed to preload model {model_id}"
            self._log.error(msg)
            raise ui.UserError(msg) from e

    def _load_discogs_embedding_model_from_path(
        self,
        path: str,
        output: str = "PartitionedCall:1",
    ) -> TensorflowPredictEffnetDiscogs:
        """Load a TensorFlow Discogs embedding model.

        Args:
        ----
            path: Path to the model file
            output: Output node name for the model

        Returns:
        -------
            Loaded TensorFlow model

        Raises:
        ------
            ModelLoadError: If model cannot be loaded

        """
        try:
            model = TensorflowPredictEffnetDiscogs(graphFilename=path, output=output)
            self._log.debug(f"Loaded Discogs embedding model: {path}")
        except Exception as e:
            msg = f"Failed to load Discogs embedding model at {path} using {output=}"
            self._log.error(msg)
            raise ModelLoadError(msg) from e

        return model

    def _load_model_from_path(
        self, path: str, model_input: str = "model/Placeholder", model_output: str = "model/Softmax"
    ) -> TensorflowPredict2D:
        """Load a standard TensorFlow model.

        Args:
        ----
            path: Path to the model file
            model_input: Input node name for the model
            model_output: Output node name for the model

        Returns:
        -------
            Loaded TensorFlow model

        Raises:
        ------
            ModelLoadError: If model cannot be loaded

        """
        try:
            model = TensorflowPredict2D(graphFilename=path, input=model_input, output=model_output)
            self._log.debug(f"Loaded model: {path}")
        except Exception as e:
            msg = f"Failed to load model {path} using {model_input=}, {model_output=}"
            self._log.error(msg)
            raise ModelLoadError(msg) from e

        return model

    def get_model(self, category: str, model_name: str) -> TensorflowPredict2D:
        """Get a loaded model by category and name.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            Loaded model if available, None otherwise

        """
        model_id = f"{category}.{model_name}"
        return self._models.get(model_id)

    def get_embedding_model(
        self,
        category: str,
        model_name: str,
    ) -> Union[TensorflowPredict2D, TensorflowPredictEffnetDiscogs, None]:
        """Get the associated embedding model for a specific model.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            Associated embedding model if available, None otherwise

        """
        model_id = f"{category}.{model_name}"
        config = self._model_configs.get(model_id)

        if not config or not config.has_embedding_reference():
            return None

        # Return the actual embedding model
        embedding_id = f"embeddings.{config.embedding}"
        return self._embeddings.get(embedding_id)

    def get_embedding_model_by_name(
        self, embedding_name: str
    ) -> Union[TensorflowPredict2D, TensorflowPredictEffnetDiscogs, None]:
        """Get an embedding model by name.

        Args:
        ----
            embedding_name: Name of the embedding model

        Returns:
        -------
            Embedding model if available, None otherwise

        """
        embedding_id = f"embeddings.{embedding_name}"
        return self._embeddings.get(embedding_id)

    def get_model_config(self, category: str, model_name: str) -> Optional[ModelConfiguration]:
        """Get the configuration for a specific model.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            ModelConfiguration if configured, None otherwise

        """
        model_id = f"{category}.{model_name}"
        return self._model_configs.get(model_id)

    def has_model(self, category: str, model_name: str) -> bool:
        """Check if a model is loaded.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            True if the model is loaded, False otherwise

        """
        model_id = f"{category}.{model_name}"
        return model_id in self._models

    def has_embedding_model(self, category: str, model_name: str) -> bool:
        """Check if the model has an associated embedding model.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            True if the model has an associated embedding model, False otherwise

        """
        # Check if model has embedding reference
        config = self.get_model_config(category, model_name)
        if not config or not config.has_embedding_reference():
            return False

        # Check if referenced embedding model exists
        embedding_id = f"embeddings.{config.embedding}"
        return embedding_id in self._embeddings

    def get_class_names(self, category: str, model_name: str) -> list[str]:
        """Get class names for a classification model.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            List of class names or empty list if not available

        """
        config = self.get_model_config(category, model_name)
        if not config:
            return []

        return config.get_class_names()

    def get_sample_rate(self, category: str, model_name: str) -> int:
        """Get the recommended sample rate for a model.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            Recommended sample rate or default (44100)

        """
        config = self.get_model_config(category, model_name)
        if not config or not config.sample_rate:
            return 44100

        return config.sample_rate
