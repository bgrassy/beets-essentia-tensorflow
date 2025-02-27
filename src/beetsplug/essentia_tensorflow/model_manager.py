"""Model management implementation for Essentia plugin."""

import logging
from pathlib import Path
from typing import Any

from beets import ui
from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs


class ModelLoadError(Exception):
    """Raised when a model fails to load."""


class ModelConfiguration:
    """Represents a model configuration with paths and output settings."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize model configuration.

        Args:
        ----
            config: Model configuration dictionary

        """
        self.model_path = str(Path(config.get("model_path", "")).expanduser()) if config.get("model_path") else None
        self.embedding = config.get("embedding")  # Name of embedding model, not path
        self.model_output = config.get("model_output", "model/Softmax")

    def is_valid(self) -> bool:
        """Check if model configuration has at least a model path."""
        return bool(self.model_path)

    def has_embedding_reference(self) -> bool:
        """Check if model configuration references an embedding model."""
        return bool(self.embedding)


class ModelManager:
    """Manages loading and validation of TensorFlow models."""

    def __init__(self, plugin_config: dict[str, Any]) -> None:
        """Initialize the model manager and preload all configured models.

        Args:
        ----
            plugin_config: Plugin configuration dictionary

        """
        self._config = plugin_config
        self._log = logging.getLogger("beets.essentia.models")
        self._models: dict[str, Any] = {}  # Store loaded models
        self._embeddings: dict[str, Any] = {}  # Store loaded embedding models

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

    def _validate_and_get_embedding_models(self, embeddings_config: dict[str, Any]) -> set[str]:
        available_embeddings = set()
        for model_name, model_config in embeddings_config.items():
            if not model_config or not model_config.get("model_path"):
                continue

            model_path = Path(model_config["model_path"]).expanduser()
            if not model_path.exists():
                msg = f"Embedding model path not found: {model_path} for embeddings.{model_name}"
                raise ui.UserError(msg)

            # Add to available embeddings
            available_embeddings.add(model_name)

        return available_embeddings

    def _validate_model_config(
        self,
        category: str,
        category_model_configs: dict[str, Any],
        available_embeddings: set[str],
    ) -> None:
        """Validate all model configurations of a specific model category."""
        for model_name, model_config in category_model_configs.items():
            if not model_config:
                continue

            # Validate model path
            if "model_path" in model_config:
                model_path = Path(model_config["model_path"]).expanduser()
                if not model_path.exists():
                    msg = f"Model path not found: {model_path} for {category}.{model_name}"
                    raise ui.UserError(msg)

            # Validate embedding reference
            if "embedding" in model_config:
                embedding_name = model_config["embedding"]
                if embedding_name not in available_embeddings:
                    msg = f"Referenced embedding model '{embedding_name}' not found for {category}.{model_name}"
                    raise ui.UserError(msg)

    def _preload_models(self) -> None:
        """Preload all configured models."""
        models_config = self._config["models"]

        # First load all embedding models
        if "embeddings" in models_config:
            for model_name, model_config in models_config["embeddings"].items():
                if not model_config or not model_config.get("model_path"):
                    continue

                self._load_embedding_model(
                    model_name,
                    model_config,
                )

        # Now load all other models
        for category in models_config:
            if category == "embeddings":
                continue  # Already loaded

            for model_name, model_config in models_config[category].items():
                model_id = f"{category}.{model_name}"

                # Skip if no config or model_path
                if not model_config or not model_config.get("model_path"):
                    msg = "Skipping model {model_name} because model path or config missing"
                    self._log.info(msg)
                    continue

                self._load_model(model_config, model_id)

    def _load_embedding_model(self, model_name: str, model_config: dict[str, Any]) -> None:
        """Load embedding model of given name from config."""
        embedding_id = f"embeddings.{model_name}"
        model_path = str(Path(model_config["model_path"]).expanduser())

        try:
            # Special case for discogs embedding model
            if model_name == "discogs":
                model_output = model_config.get("model_output", "PartitionedCall:1")
                self._embeddings[embedding_id] = self._load_discogs_embedding_model_from_path(model_path, model_output)
            else:
                # TODO: Add other embedding types
                model_output = model_config.get("model_output", "model/Softmax")
                self._embeddings[embedding_id] = self._load_model_from_path(model_path, model_output)

            self._log.info(f"Preloaded embedding model: {embedding_id}")
        except Exception as e:
            msg = f"Failed to preload embedding model {embedding_id}"
            self._log.error(msg)
            raise ui.UserError(msg) from e

    def _load_model(self, model_config: dict[str, Any], model_id: str) -> None:
        """Load generic Tensorflow model of given name from config."""
        model_path = str(Path(model_config["model_path"]).expanduser())
        model_output = model_config.get("model_output", "model/Softmax")

        try:
            # Load the model
            self._models[model_id] = self._load_model_from_path(model_path, model_output)
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

    def _load_model_from_path(self, path: str, model_output: str = "model/Softmax") -> TensorflowPredict2D:
        """Load a standard TensorFlow model.

        Args:
        ----
            path: Path to the model file
            model_output: Output node name for the model

        Returns:
        -------
            Loaded TensorFlow model

        Raises:
        ------
            ModelLoadError: If model cannot be loaded

        """
        try:
            model = TensorflowPredict2D(graphFilename=path, output=model_output)
            self._log.debug(f"Loaded model: {path}")
        except Exception as e:
            msg = f"Failed to load model {path}"
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
    ) -> TensorflowPredict2D | TensorflowPredictEffnetDiscogs:
        """Get the associated embedding model for a specific model.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            Associated embedding model if available, None otherwise

        """
        # Get embedding model name from configuration
        model_config = self.get_model_config(category, model_name)
        if not model_config or not model_config.has_embedding_reference():
            return None

        # Return the actual embedding model
        embedding_id = f"embeddings.{model_config.embedding}"
        return self._embeddings.get(embedding_id)

    def get_embedding_model_by_name(self, embedding_name: str) -> TensorflowPredict2D | TensorflowPredictEffnetDiscogs:
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

    def get_model_config(self, category: str, model_name: str) -> ModelConfiguration | None:
        """Get the configuration for a specific model.

        Args:
        ----
            category: Model category (e.g., 'classification')
            model_name: Model name (e.g., 'genre')

        Returns:
        -------
            ModelConfiguration if configured, None otherwise

        """
        model_config = self._config["models"].get(category, {}).get(model_name, None)
        if not model_config:
            return None
        return ModelConfiguration(model_config)

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
        model_config = self.get_model_config(category, model_name)
        if not model_config or not model_config.has_embedding_reference():
            return False

        # Check if referenced embedding model exists
        embedding_id = f"embeddings.{model_config.embedding}"
        return embedding_id in self._embeddings
