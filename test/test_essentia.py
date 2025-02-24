"""Essentia tests."""
import logging

from beets.plugins import BeetsPlugin
from beetsplug.essentia import EssentiaPlugin
from beets.test.helper import PluginTestCase, capture_log

log = logging.getLogger("beets")
log.propagate = True
log.setLevel(logging.DEBUG)


class TestEssentiaPlugin(PluginTestCase):
    """Essentia test functions."""

    plugin = "essentia"

    def setUp(self) -> None:
        """Set up the test class."""
        super().setUp()

    def test_plugin_load(self) -> None:
        """Test that plugin loads successfully."""
        plugin = EssentiaPlugin()
        assert isinstance(plugin, BeetsPlugin)
        assert plugin._log.name == "beets.essentia"  # noqa: SLF001


    def test_default_config(self) -> None:
        """Test that default configuration loads correctly."""
        plugin = EssentiaPlugin()
        assert plugin.config["auto"].get(bool) is False
        assert plugin.config["write"].get(bool) is True
        assert plugin.config["threads"].get(int) == 1
        assert plugin.config["force"].get(bool) is False
        assert plugin.config["quiet"].get(bool) is False


    def test_commands_registration(self) -> None:
        """Test that commands are registered properly."""
        plugin = EssentiaPlugin()
        commands = plugin.commands()
        assert len(commands) == 1

        command = commands[0]
        assert command.name == "essentia"
        assert command.aliases == ["ess"]
        assert callable(command.func)


    def test_logging_setup(self) -> None:
        """Test that logging is properly configured."""
        with capture_log() as logs:
            EssentiaPlugin()
            assert "essentia: Plugin configuration validated successfully" in logs


    def test_command_execution(self) -> None:
        """Test basic command execution."""
        plugin = EssentiaPlugin()
        command = plugin.commands()[0]
        with capture_log() as logs:
            command.func(None, None, None)

        assert "essentia: Essentia analysis starting..." in logs
        assert "essentia: Essentia analysis complete" in logs


    def test_handle_album(self) -> None:
        """Test that handler methods exist and log properly."""
        plugin = EssentiaPlugin()
        with capture_log() as logs:
            plugin.handle_album(None, None, None)

        assert "essentia: Album processing requested" in logs


    def test_handle_item(self) -> None:
        """Test that handler methods exist and log properly."""
        plugin = EssentiaPlugin()
        with capture_log() as logs:
            plugin.handle_item(None, None, None)

        assert "essentia: Item processing requested" in logs
