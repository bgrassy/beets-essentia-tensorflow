"""Integration tests for the Essentia plugin."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import json

import numpy as np
from beets.library import Item
from beets.test.helper import TestHelper, capture_log

from beetsplug.essentia_tensorflow import EssentiaPlugin
from beetsplug.essentia_tensorflow.model_manager import ModelManager
from beetsplug.essentia_tensorflow.audio_processor import AudioProcessor


class TestPluginIntegration(TestHelper):
    """Integration tests for the Essentia plugin."""

    def setUp(self) -> None:
        """Set up the test environment."""
        super().setUp()

        # Create a test library
        self.setup_beets()

        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = self.tmp_dir / "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # Create mock model files
        self.create_mock_model_files()

        # Configure plugin
        self.config_plugin()

        # Create the plugin instance
        self.plugin = EssentiaPlugin()

        # Mock the model manager and audio processor
        self.mock_models()

    def tearDown(self) -> None:
        """Clean up after the test."""
        super().tearDown()

        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)

    def create_mock_model_files(self) -> None:
        """Create empty files for mock models."""
        model_files = [
            "discogs_model",
            "musicnn_model",
            "genre_model",
            "mood_model",
            "tempo_model",
            "key_model",
        ]

        for model_file in model_files:
            with open(os.path.join(self.model_dir, model_file), "w") as f:
                f.write("# Mock model file")

    def config_plugin(self) -> None:
        """Configure the plugin for testing."""
        # Use self.config, which is the beets configuration
        self.config["essentia"] = {
            "auto": False,
            "dry_run": False,
            "write": True,
            "threads": 1,
            "force": False,
            "quiet": False,
            "models": {
                "embeddings": {
                    "discogs": os.path.join(self.model_dir, "discogs_model"),
                    "musicnn": os.path.join(self.model_dir, "musicnn_model"),
                },
                "classification": {
                    "genre": {
                        "model_path": os.path.join(self.model_dir, "genre_model"),
                        "embedding": "discogs",
                    },
                    "mood": {
                        "model_path": os.path.join(self.model_dir, "mood_model"),
                        "embedding": "musicnn",
                    },
                },
                "rhythm": {
                    "tempocnn": {
                        "model_path": os.path.join(self.model_dir, "tempo_model"),
                    },
                },
                "harmony": {
                    "key": {
                        "model_path": os.path.join(self.model_dir, "key_model"),
                    },
                },
            },
            "storage": {
                "tags": {
                    "write": True,
                    "update_existing": False,
                    "formats": {
                        "id3": True,
                        "vorbis": True,
                        "mp4": True,
                        "asf": True,
                    },
                    "fields": {
                        "bpm": True,
                        "key": True,
                        "genre": True,
                        "mood": True,
                        "dance": False,
                        "voice": False,
                    },
                },
                "database": {
                    "store_probabilities": True,
                    "beat_resolution": 0.001,
                    "chord_format": "simple",
                },
            },
        }

    def mock_models(self):
        """Mock the model manager and audio processor."""
        # Create mock model manager
        mock_model_manager = MagicMock(spec=ModelManager)

        # Mock has_model to return True for specific models
        def has_model_side_effect(category, model_name):
            if category == "classification" and model_name in ["genre", "mood"]:
                return True
            if category == "rhythm" and model_name in ["tempocnn"]:
                return True
            if category == "harmony" and model_name in ["key"]:
                return True
            return False

        mock_model_manager.has_model.side_effect = has_model_side_effect

        # Mock get_model to return models
        def get_model_side_effect(category, model_name):
            model = MagicMock()
            if category == "classification":
                if model_name == "genre":
                    model.return_value = np.array([0.1, 0.7, 0.2])
                elif model_name == "mood":
                    model.return_value = np.array([0.3, 0.6, 0.1])
            elif category == "rhythm" and model_name == "tempocnn":
                model.return_value = np.array([120.5])
            elif category == "harmony" and model_name == "key":
                model.return_value = np.array([0.05] * 12 + [0.8, 0.05] * 6)
            return model

        mock_model_manager.get_model.side_effect = get_model_side_effect

        # Mock audio processor
        mock_audio_processor = MagicMock(spec=AudioProcessor)

        # Mock process_file to return test results
        def process_file_side_effect(file_path):
            return {
                "genre": {
                    "feature_type": "genre",
                    "model_name": "genre",
                    "value": "rock",
                    "confidence": 0.7,
                    "probabilities": [0.1, 0.7, 0.2],
                },
                "mood": {
                    "feature_type": "mood",
                    "model_name": "mood",
                    "value": "happy",
                    "confidence": 0.6,
                    "probabilities": [0.3, 0.6, 0.1],
                },
                "bpm": {
                    "feature_type": "bpm",
                    "model_name": "tempocnn",
                    "value": 120.5,
                },
                "key": {
                    "feature_type": "key",
                    "model_name": "key",
                    "value": "C",
                    "confidence": 0.8,
                },
            }

        mock_audio_processor.process_file.side_effect = process_file_side_effect

        # Patch the plugin's methods to return our mocks
        self.plugin._get_or_create_model_manager = MagicMock(return_value=mock_model_manager)
        self.plugin._get_or_create_audio_processor = MagicMock(return_value=mock_audio_processor)

    def create_test_mp3(self, path):
        """Create a test MP3 file at the given path."""
        # This just creates an empty file for testing
        with open(path, "wb") as f:
            f.write(b"")

    def test_process_items(self):
        """Test processing items through the plugin."""
        # Create a test item
        test_path = os.path.join(self.temp_dir, "test.mp3")
        self.create_test_mp3(test_path)

        item = Item(path=test_path, title="Test Track")
        item.add(self.lib)

        # Process the item
        with capture_log() as logs:
            self.plugin._process_items(self.lib, [item])

        # Verify that the processor was called
        self.plugin._get_or_create_audio_processor.return_value.process_file.assert_called_once_with(test_path)

        # Refresh the item from the database to get updated values
        item.load()

        # Check that the fields were updated
        assert item.essentia_genre == "rock"
        assert json.loads(item.essentia_genre_probs) == [0.1, 0.7, 0.2]
        assert item.essentia_mood == "happy"
        assert json.loads(item.essentia_mood_probs) == [0.3, 0.6, 0.1]
        assert item.essentia_bpm == 120.5
        assert item.essentia_key == "C"
        assert item.essentia_key_prob == 0.8

        # Verify log messages
        assert "Processing" in "".join(logs)
        assert "Updated item" in "".join(logs)

    def test_skip_already_processed(self):
        """Test that already processed items are skipped unless forced."""
        # Create a test item
        test_path = os.path.join(self.temp_dir, "test.mp3")
        self.create_test_mp3(test_path)

        item = Item(path=test_path, title="Test Track", essentia_genre="existing")
        item.add(self.lib)

        # Process without force
        with capture_log() as logs:
            self.plugin._process_items(self.lib, [item])

        # Verify that the processor was not called
        self.plugin._get_or_create_audio_processor.return_value.process_file.assert_not_called()

        # Verify log messages
        assert "Skipping" in "".join(logs)

        # Now process with force
        self.plugin.config["force"] = True

        with capture_log() as logs:
            self.plugin._process_items(self.lib, [item])

        # Verify that the processor was called
        self.plugin._get_or_create_audio_processor.return_value.process_file.assert_called_once_with(test_path)

    def test_dry_run(self):
        """Test dry run mode."""
        # Create a test item
        test_path = os.path.join(self.temp_dir, "test.mp3")
        self.create_test_mp3(test_path)

        item = Item(path=test_path, title="Test Track")
        item.add(self.lib)

        # Enable dry run
        self.plugin.config["dry_run"] = True

        # Process the item
        with capture_log() as logs:
            self.plugin._process_items(self.lib, [item])

        # Verify that the processor was called
        self.plugin._get_or_create_audio_processor.return_value.process_file.assert_called_once_with(test_path)

        # Refresh the item from the database
        item.load()

        # Check that the fields were not updated
        assert not hasattr(item, "essentia_genre") or item.essentia_genre is None

        # Verify log messages
        assert "Dry run" in "".join(logs)

    def test_command_integration(self):
        """Test the essentia command."""
        # Create a test item
        test_path = os.path.join(self.temp_dir, "test.mp3")
        self.create_test_mp3(test_path)

        item = Item(path=test_path, title="Test Track")
        item.add(self.lib)

        # Get the command
        commands = self.plugin.commands()
        essentia_cmd = commands[0]

        # Run the command
        with capture_log() as logs:
            essentia_cmd.func(self.lib, None, [])

        # Check log messages
        assert "Essentia analysis starting" in "".join(logs)
        assert "Essentia analysis complete" in "".join(logs)

        # Verify that the processor was called
        self.plugin._get_or_create_audio_processor.return_value.process_file.assert_called_once_with(test_path)

        # Refresh the item from the database
        item.load()

        # Check that the fields were updated
        assert item.essentia_genre == "rock"
        assert item.essentia_bpm == 120.5

    def test_album_processing(self):
        """Test processing albums."""
        # Create test items for an album
        album_dir = os.path.join(self.temp_dir, "album")
        os.makedirs(album_dir, exist_ok=True)

        test_path1 = os.path.join(album_dir, "track1.mp3")
        test_path2 = os.path.join(album_dir, "track2.mp3")

        self.create_test_mp3(test_path1)
        self.create_test_mp3(test_path2)

        # Add items to the library as part of an album
        album = self.add_album_fixture(albumartist="Artist", album="Album")
        item1 = self.add_item_fixture(title="Track 1", album="Album", path=test_path1)
        item2 = self.add_item_fixture(title="Track 2", album="Album", path=test_path2)

        # Process the album
        with capture_log() as logs:
            self.plugin.handle_album(self.lib, None, ["Album"])

        # Verify processor calls
        assert self.plugin._get_or_create_audio_processor.return_value.process_file.call_count == 2

        # Check log messages
        assert "Processing" in "".join(logs)
        assert "from" in "".join(logs)

    def test_tag_writing(self):
        """Test tag writing."""
        # Create a test item
        test_path = os.path.join(self.temp_dir, "test.mp3")
        self.create_test_mp3(test_path)

        item = Item(path=test_path, title="Test Track")
        item.add(self.lib)

        # Create results
        results = {
            "genre": {
                "feature_type": "genre",
                "model_name": "genre",
                "value": "rock",
                "confidence": 0.7,
            },
            "bpm": {
                "feature_type": "bpm",
                "model_name": "tempocnn",
                "value": 120.5,
            },
            "key": {
                "feature_type": "key",
                "model_name": "key",
                "value": "C",
                "confidence": 0.8,
            },
        }

        # Mock write_tags method
        with patch.object(self.plugin, "_write_tags") as mock_write_tags:
            # Process the item
            self.plugin._process_items(self.lib, [item])

            # Verify write_tags was called
            mock_write_tags.assert_called_once()

            # Verify the format was checked
            assert "test.mp3" in mock_write_tags.call_args[0][0].path

    def test_format_detection(self):
        """Test format detection for tag writing."""
        # Test various formats
        formats = {
            "mp3": True,
            "flac": True,
            "m4a": True,
            "wma": True,
            "txt": False,
        }

        for ext, expected in formats.items():
            test_path = os.path.join(self.temp_dir, f"test.{ext}")
            with open(test_path, "w") as f:
                f.write("")

            item = Item(path=test_path, title=f"Test {ext}")

            result = self.plugin._should_write_tags_for_format(item)
            assert result == expected, f"Expected {expected} for {ext}, got {result}"
