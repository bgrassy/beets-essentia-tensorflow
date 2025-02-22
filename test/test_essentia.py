# test/test_essentia.py

import logging

import pytest
from beets import ui
from beets.plugins import BeetsPlugin
from beetsplug.essentia import EssentiaPlugin

log = logging.getLogger("beets")
log.propagate = True
log.setLevel(logging.DEBUG)


def test_plugin_load():
    """Test that plugin loads successfully."""
    plugin = EssentiaPlugin()
    assert isinstance(plugin, BeetsPlugin)
    assert plugin._log.name == "beets.essentia"


def test_default_config():
    """Test that default configuration loads correctly."""
    plugin = EssentiaPlugin()
    assert plugin.config["auto"].get() is False
    assert plugin.config["write"].get() is True
    assert plugin.config["threads"].get() == 1
    assert plugin.config["force"].get() is False
    assert plugin.config["quiet"].get() is False


def test_commands_registration():
    """Test that commands are registered properly."""
    plugin = EssentiaPlugin()
    commands = plugin.commands()
    assert len(commands) == 1

    command = commands[0]
    assert command.name == "essentia"
    assert command.aliases == ["ess"]
    assert callable(command.func)


def test_invalid_threads_config():
    """Test that invalid thread count raises error."""
    plugin = EssentiaPlugin()
    plugin.config["threads"] = 0

    with pytest.raises(ui.UserError):
        plugin._validate_config()


def test_logging_setup(caplog):
    """Test that logging is properly configured."""
    plugin = EssentiaPlugin()
    with caplog.at_level(logging.DEBUG):
        plugin._setup_plugin()

    logs = caplog.text

    assert "essentia: Plugin configuration validated successfully" in logs


def test_command_execution(caplog):
    """Test basic command execution."""
    plugin = EssentiaPlugin()
    command = plugin.commands()[0]
    with caplog.at_level(logging.INFO):
        command.func(None, None, None)

    assert "essentia: Essentia analysis starting..." in caplog.text
    assert "essentia: Essentia analysis complete" in caplog.text


@pytest.mark.parametrize("method_name", ["handle_album", "handle_item"])
def test_handler_methods(method_name, caplog):
    """Test that handler methods exist and log properly."""
    plugin = EssentiaPlugin()
    with caplog.at_level(logging.DEBUG):
        method = getattr(plugin, method_name)
        method(None, None, None)

    assert "processing requested" in caplog.text
