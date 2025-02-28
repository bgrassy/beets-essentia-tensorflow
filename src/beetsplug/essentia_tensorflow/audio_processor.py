"""Audio processing implementation for Essentia plugin."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from beets.util import bytestring_path, syspath
from essentia.standard import MonoLoader

from beetsplug.essentia_tensorflow.model_manager import ModelManager
from IPython import embed


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""


@dataclass
class FeatureExtractionResult:
    """Container for storing feature extraction results."""

    feature_type: str
    model_name: str
    values: list[float] | list[list[float]] | dict[str, Any]
    probabilities: Optional[list[float]] = None
    labels: Optional[list[str]] = None
    confidence: Optional[float] = None
    timestamp: Optional[float] = None
    metadata: dict[str, Any] | None = None

    def get_top_result(self) -> Tuple[str, float]:
        """Get the top result label and its probability.

        Returns
        -------
            Tuple of (label, probability)

        Raises
        ------
            ValueError: If labels or probabilities are not available

        """
        if not self.labels or not self.probabilities:
            raise ValueError("Labels and probabilities must be available to get top result")

        top_idx = np.argmax(self.probabilities)
        return self.labels[top_idx], self.probabilities[top_idx]

    def get_top_n_results(self, n: int = 3) -> list[Tuple[str, float]]:
        """Get the top N result labels and their probabilities.

        Args:
        ----
            n: Number of top results to return

        Returns:
        -------
            list of (label, probability) tuples

        Raises:
        ------
            ValueError: If labels or probabilities are not available

        """
        if not self.labels or not self.probabilities:
            raise ValueError("Labels and probabilities must be available to get top results")

        # Convert to numpy array for easier manipulation
        probs = np.array(self.probabilities)
        # Get indices of top N values
        top_indices = np.argsort(probs)[-n:][::-1]
        # Return list of (label, probability) tuples
        return [(self.labels[i], probs[i]) for i in top_indices]

    def format_for_database(self) -> dict[str, Any]:
        """Format the result for storage in the database.

        Returns
        -------
            dictionary formatted for database storage

        """
        result = {"feature_type": self.feature_type, "model_name": self.model_name}

        # Handle different result types
        if self.feature_type in ("genre", "style", "mood", "voice_instrumental"):
            # For classification results
            result["value"], result["confidence"] = self.get_top_result()
            if self.probabilities:
                result["probabilities"] = self.probabilities

        elif self.feature_type in ("bpm", "key", "chords"):
            # For rhythm and harmony results
            if self.feature_type == "bpm":
                result["value"] = float(self.values[0])
            elif self.feature_type == "key":
                result["value"] = self.labels[np.argmax(self.probabilities)] if self.labels else str(self.values[0])
                result["confidence"] = float(np.max(self.probabilities)) if self.probabilities else None
            elif self.feature_type == "chords":
                result["value"] = self.values
        else:
            # For other types, just store values directly
            result["value"] = self.values

        # Add metadata if available
        if self.metadata:
            result["metadata"] = self.metadata

        return result


class AudioProcessor:
    """Handles audio processing and feature extraction using Essentia models."""

    # Known feature types and their expected output formats
    FEATURE_TYPES = {
        "genre": {"classification": ["genre"]},
        "style": {"classification": ["style"]},
        "mood": {"classification": ["mood"]},
        "voice_instrumental": {"classification": ["voice_instrumental"]},
        "danceability": {"classification": ["danceability"]},
        "bpm": {"rhythm": ["tempocnn"]},
        "beats": {"rhythm": ["beats"]},
        "key": {"harmony": ["key"]},
        "chords": {"harmony": ["chords"]},
    }

    def __init__(self, model_manager: ModelManager, config: dict[str, Any]) -> None:
        """Initialize the audio processor.

        Args:
        ----
            model_manager: The model manager containing loaded models
            config: Plugin configuration dictionary

        """
        self._model_manager = model_manager
        self._config = config
        self._log = logging.getLogger("beets.essentia.processor")

        # Validate available models
        self._available_features = self._get_available_features()
        self._log.info(f"Available features for extraction: {', '.join(self._available_features)}")

    def _get_available_features(self) -> list[str]:
        """Determine which features can be extracted based on available models.

        Returns
        -------
            list of available feature types

        """
        available = []

        for feature, categories in self.FEATURE_TYPES.items():
            for category, models in categories.items():
                # Check if all required models for this feature are available
                if all(self._model_manager.has_model(category, model) for model in models):
                    available.append(feature)
                    break

        return available

    def process_file(self, file_path: str) -> dict[str, Any]:
        """Process an audio file and extract all available features.

        Args:
        ----
            file_path: Path to the audio file

        Returns:
        -------
            dictionary containing all extracted features

        Raises:
        ------
            AudioProcessingError: If processing fails

        """
        try:
            self._log.info(f"Processing file: {file_path}")

            # Validate file exists
            path = Path(file_path)
            if not path.exists():
                msg = f"File does not exist: {file_path}"
                raise AudioProcessingError(msg)

            # Extract all available features
            results = {}
            for feature in self._available_features:
                try:
                    result = self._extract_feature(file_path, feature)
                    if result:
                        results[feature] = result.format_for_database()
                except Exception as e:
                    self._log.error(f"Error extracting {feature} from {file_path}: {e}", exc_info=True)
                    # Continue with other features even if one fails

            if not results:
                msg = f"No features could be extracted from {file_path}"
                raise AudioProcessingError(msg)

            return results

        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                msg = f"Error processing file {file_path}: {e}"
                self._log.error(msg)
                raise AudioProcessingError(msg) from e
            raise

    def _extract_feature(self, file_path: str, feature_type: str) -> Optional[FeatureExtractionResult]:
        """Extract a specific feature from an audio file.

        Args:
        ----
            file_path: Path to the audio file
            feature_type: Type of feature to extract

        Returns:
        -------
            FeatureExtractionResult or None if extraction fails

        """
        if feature_type not in self.FEATURE_TYPES:
            self._log.warning(f"Unknown feature type: {feature_type}")
            return None

        # Get model category and name
        for category, models in self.FEATURE_TYPES[feature_type].items():
            model_name = models[0]  # Take first model as primary
            if not self._model_manager.has_model(category, model_name):
                self._log.warning(f"Model {category}.{model_name} not available")
                return None

            # Get processing parameters for this model from the model config
            config = self._model_manager.get_model_config(category, model_name)
            sample_rate = self._model_manager.get_sample_rate(category, model_name)

            # Default parameters if not available in config
            params = {"sample_rate": sample_rate, "resample_quality": 4, "segment_duration": 10}

            # Load audio
            audio = self._load_audio(file_path, params["sample_rate"], params["resample_quality"])
            if audio is None:
                self._log.error(f"Failed to load audio from {file_path}")
                return None

            # Extract feature
            if feature_type in ("genre", "style", "mood", "voice_instrumental", "danceability"):
                return self._extract_classification_feature(audio, category, model_name, feature_type, params)
            if feature_type == "bpm":
                return self._extract_bpm(audio, category, model_name, params)
            if feature_type == "key":
                return self._extract_key(audio, category, model_name, params)
            if feature_type == "chords":
                return self._extract_chords(audio, category, model_name, params)
            if feature_type == "beats":
                return self._extract_beats(audio, category, model_name, params)

        return None

    def _load_audio(self, file_path: str, sample_rate: int = 44100, resample_quality: int = 4) -> Optional[np.ndarray]:
        """Load audio file using Essentia.

        Args:
        ----
            file_path: Path to the audio file
            sample_rate: Sample rate to load the audio at
            resample_quality: Quality of resampling

        Returns:
        -------
            Numpy array of audio samples or None if loading fails

        """
        try:
            # Use MonoLoader for consistent audio loading
            loader = MonoLoader(
                filename=syspath(bytestring_path(file_path)),
                sampleRate=sample_rate,
                resampleQuality=resample_quality,
            )
            audio = loader()

            if len(audio) == 0:
                self._log.error(f"Empty audio loaded from {file_path}")
                return None

            return audio
        except Exception as e:
            self._log.error(f"Error loading audio from {file_path}: {e}")
            return None

    def _extract_classification_feature(
        self,
        audio: np.ndarray,
        category: str,
        model_name: str,
        feature_type: str,
        params: dict[str, Any],
    ) -> Optional[FeatureExtractionResult]:
        """Extract classification features (genre, mood, etc.).

        Args:
        ----
            audio: Audio samples
            category: Model category
            model_name: Model name
            feature_type: Type of feature to extract
            params: Processing parameters

        Returns:
        -------
            FeatureExtractionResult or None if extraction fails
        """

        try:
            model = self._model_manager.get_model(category, model_name)
            if model is None:
                self._log.error(f"Model {category}.{model_name} not found")
                return None
            # Get associated embedding model if required
            embedding_model = self._model_manager.get_embedding_model(category, model_name)

            # Process audio through embedding model if needed
            if embedding_model is not None:
                # Apply embedding model
                embedding = embedding_model(audio)

                # Apply classification model to embedding
                predictions = model(embedding)
            else:
                # Direct prediction
                predictions = model(audio)

            # Process predictions
            probabilities = predictions.mean(axis=0).tolist()

            # Get labels from model metadata
            labels = self._model_manager.get_class_names(category, model_name)

            # If no labels are available from metadata, generate placeholder labels
            if not labels:
                labels = [f"{feature_type}_{i}" for i in range(len(probabilities))]

            return FeatureExtractionResult(
                feature_type=feature_type,
                model_name=model_name,
                values=predictions.tolist(),
                probabilities=probabilities,
                labels=labels,
            )

        except Exception as e:
            self._log.error(f"Error extracting {feature_type}: {e}")
            return None

    def _extract_bpm(
        self,
        audio: np.ndarray,
        category: str,
        model_name: str,
        params: dict[str, Any],
    ) -> Optional[FeatureExtractionResult]:
        """Extract BPM (beats per minute).

        Args:
        ----
            audio: Audio samples
            category: Model category
            model_name: Model name
            params: Processing parameters

        Returns:
        -------
            FeatureExtractionResult or None if extraction fails

        """
        try:
            model = self._model_manager.get_model(category, model_name)
            if model is None:
                self._log.error(f"Model {category}.{model_name} not found")
                return None

            # Process audio through the model
            predictions = model(audio)

            # Extract BPM value
            # Assuming predictions contains BPM values with confidence
            # In a real implementation, the interpretation would depend on the specific model
            bpm_value = float(predictions[0])

            return FeatureExtractionResult(
                feature_type="bpm",
                model_name=model_name,
                values=[bpm_value],
                confidence=0.9,  # Placeholder confidence
                metadata={"segment_duration": params["segment_duration"]},
            )

        except Exception as e:
            self._log.error(f"Error extracting BPM: {e}")
            return None

    def _extract_key(
        self,
        audio: np.ndarray,
        category: str,
        model_name: str,
        params: dict[str, Any],
    ) -> Optional[FeatureExtractionResult]:
        """Extract musical key.

        Args:
        ----
            audio: Audio samples
            category: Model category
            model_name: Model name
            params: Processing parameters

        Returns:
        -------
            FeatureExtractionResult or None if extraction fails

        """
        try:
            model = self._model_manager.get_model(category, model_name)
            if model is None:
                self._log.error(f"Model {category}.{model_name} not found")
                return None

            # Process audio through the model
            predictions = model(audio)

            # Get key labels from model metadata or use default
            key_labels = self._model_manager.get_class_names(category, model_name)

            # If no labels available, use default key labels
            if not key_labels:
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

            # Extract key prediction
            # Assuming predictions is a vector of probabilities for each key
            probabilities = predictions.tolist()

            return FeatureExtractionResult(
                feature_type="key",
                model_name=model_name,
                values=predictions.tolist(),
                probabilities=probabilities,
                labels=key_labels,
                metadata={"segment_duration": params["segment_duration"]},
            )

        except Exception as e:
            self._log.error(f"Error extracting key: {e}")
            return None

    def _extract_chords(
        self,
        audio: np.ndarray,
        category: str,
        model_name: str,
        params: dict[str, Any],
    ) -> Optional[FeatureExtractionResult]:
        """Extract chord progression.

        Args:
        ----
            audio: Audio samples
            category: Model category
            model_name: Model name
            params: Processing parameters

        Returns:
        -------
            FeatureExtractionResult or None if extraction fails

        """
        try:
            model = self._model_manager.get_model(category, model_name)
            if model is None:
                self._log.error(f"Model {category}.{model_name} not found")
                return None

            # Process audio through the model
            # Chord detection would typically return time-aligned chord symbols
            predictions = model(audio)

            # Format chords based on configuration
            chord_format = self._config["storage"]["database"]["chord_format"]

            # Get chord labels from model metadata if available
            chord_labels = self._model_manager.get_class_names(category, model_name)

            # In a real implementation, this would parse the model output based on its format
            # For now, we'll generate placeholder chord data
            if chord_format == "simple":
                chords = ["C", "G", "Am", "F"]  # Placeholder
            else:  # detailed
                chords = [
                    {"chord": "C", "start": 0.0, "end": 1.0},
                    {"chord": "G", "start": 1.0, "end": 2.0},
                    {"chord": "Am", "start": 2.0, "end": 3.0},
                    {"chord": "F", "start": 3.0, "end": 4.0},
                ]

            return FeatureExtractionResult(
                feature_type="chords",
                model_name=model_name,
                values=chords,
                labels=chord_labels if chord_labels else None,
                metadata={"format": chord_format},
            )

        except Exception as e:
            self._log.error(f"Error extracting chords: {e}")
            return None

    def _extract_beats(
        self,
        audio: np.ndarray,
        category: str,
        model_name: str,
        params: dict[str, Any],
    ) -> Optional[FeatureExtractionResult]:
        """Extract beat positions.

        Args:
        ----
            audio: Audio samples
            category: Model category
            model_name: Model name
            params: Processing parameters

        Returns:
        -------
            FeatureExtractionResult or None if extraction fails

        """
        try:
            model = self._model_manager.get_model(category, model_name)
            if model is None:
                self._log.error(f"Model {category}.{model_name} not found")
                return None

            # Process audio through the model
            predictions = model(audio)

            # In a real implementation, this would convert the model output to beat times
            # For now, we'll generate placeholder beat data
            beat_resolution = self._config["storage"]["database"]["beat_resolution"]
            # Generate evenly spaced beats at 120 BPM as a placeholder
            beat_interval = 60 / 120  # 120 BPM = 0.5 seconds per beat
            duration = len(audio) / params["sample_rate"]
            beat_times = [round(i * beat_interval, 3) for i in range(int(duration / beat_interval))]

            return FeatureExtractionResult(
                feature_type="beats",
                model_name=model_name,
                values=beat_times,
                metadata={"resolution": beat_resolution},
            )

        except Exception as e:
            self._log.error(f"Error extracting beats: {e}")
            return None
