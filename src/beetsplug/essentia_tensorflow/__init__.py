"""Essentia-tensorflow beets plugin."""

import logging
from pathlib import Path

from beets import ui
from beets.plugins import BeetsPlugin
from beets.util import bytestring_path, syspath


class EssentiaPlugin(BeetsPlugin):
    """Beets plugin for audio analysis using Essentia-TensorFlow.

    This plugin provides music analysis capabilities through the Essentia library,
    including genre detection, mood analysis, and other audio characteristics.
    """

    def __init__(self) -> None:
        """Initialize essentia plugin."""
        super().__init__()

        # Set up logging
        self._log = logging.getLogger("beets.essentia")

        # Define the default configuration schema
        # Basic processing options
        self.config.add(
            {
                "auto": False,
                "dry_run": False,
                "write": True,
                "threads": 1,
                "force": False,
                "quiet": False,
                "models": {
                    "embeddings": {
                        "musicnn": "",
                        "vggish": "",
                        "discogs": "",
                    },
                    "classification": {
                        "genre": "",
                        "style": "",
                        "mood": "",
                        "danceability": "",
                        "voice_instrumental": "",
                    },
                    "rhythm": {
                        "tempocnn": "",
                        "beats": "",
                    },
                    "harmony": {
                        "key": "",
                        "chords": "",
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
                            "genre": False,
                            "mood": False,
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
            },
        )

        # Model paths configuration
        self._setup_plugin()

    def _setup_plugin(self) -> None:
        """Initialize plugin components and validate configuration."""
        try:
            self._validate_config()
            self._log.debug("Plugin configuration validated successfully")
        except Exception as e:
            self._log.error(f"Error during plugin setup: {e!s}")
            raise

    def _validate_config(self) -> None:
        """Validate plugin configuration values."""
        self._validate_basic_options()
        self._validate_model_paths()
        self._validate_storage_config()

    def _validate_basic_options(self) -> None:
        """Validate basic plugin options."""
        # Validate thread count
        threads = self.config["threads"].get(int)
        if threads < 1:
            msg = "Threads must be at least 1"
            raise ui.UserError(msg)

    def _validate_model_paths(self) -> None:
        """Validate model paths if they are specified."""
        models_config = self.config["models"].get(dict)

        for category in models_config:
            for model, path in models_config[category].items():
                if path:  # Only validate if path is specified
                    full_path = Path.expanduser(bytestring_path(path))
                    if not Path.exists(syspath(full_path)):
                        msg = f"Model path not found: {full_path} for {category}.{model}"
                        raise ui.UserError(msg)

    def _validate_storage_config(self) -> None:
        """Validate storage-related configuration."""
        storage_config = self.config["storage"].get(dict)

        # Validate tag configuration
        tags_config = storage_config["tags"]
        if not isinstance(tags_config["write"], bool):
            msg = "storage.tags.write must be a boolean"
            raise ui.UserError(msg)

        # Validate database configuration
        db_config = storage_config["database"]
        if not isinstance(db_config["beat_resolution"], int | float):
            msg = "storage.database.beat_resolution must be a number"
            raise ui.UserError(msg)
        if db_config["beat_resolution"] <= 0:
            msg = "storage.database.beat_resolution must be positive"
            raise ui.UserError(msg)

        if db_config["chord_format"] not in ["simple", "detailed"]:
            msg = "storage.database.chord_format must be 'simple' or 'detailed'"
            raise ui.UserError(msg)

    def _get_model_path(self, category: str, model: str) -> str | None:
        """Get the full path for a specific model."""
        path = self.config["models"][category][model].get()
        if not path:
            return None
        return str(Path.expanduser(bytestring_path(path)))

    def commands(self) -> list[ui.Subcommand]:
        """Create and register plugin commands."""
        essentia_cmd = ui.Subcommand(
            "essentia",
            help="Extract file data using essentia",
            aliases=["ess"],
        )

        def func(lib, opts, args) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001, ARG001
            self._log.info("Essentia analysis starting...")
            # Command implementation will go here
            self._log.info("Essentia analysis complete")

        essentia_cmd.func = func
        return [essentia_cmd]

    def handle_album(self, lib, opts, args) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001, ARG002
        """Process albums from command line."""
        self._log.debug("Album processing requested")
        # Album handling implementation will go here

    def handle_item(self, lib, opts, args) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001, ARG002
        """Process individual tracks from command line."""
        self._log.debug("Item processing requested")
        # Item handling implementation will go here
