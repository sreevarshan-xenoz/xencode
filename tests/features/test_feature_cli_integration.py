"""Integration tests for the feature CLI command group."""

from click.testing import CliRunner

from xencode.features.base import FeatureStatus
from xencode.features.core.cli import FeatureCommandGroup


class MockFeature:
    """Simple mock feature for CLI integration tests."""

    version = "1.0.0"

    def get_status(self):
        return FeatureStatus.ENABLED

    @property
    def is_enabled(self):
        return True

    @property
    def is_initialized(self):
        return True


class MockFeatureClass:
    """Mock feature class used by info command."""

    __doc__ = "Mock feature used for testing."
    version = "1.0.0"


class MockFeatureManager:
    """Mock manager compatible with FeatureCommandGroup expectations."""

    def __init__(self):
        self._feature = MockFeature()

    def get_available_features(self):
        return ["mock_feature"]

    def get_feature(self, name):
        if name == "mock_feature":
            return self._feature
        return None

    def get_feature_class(self, name):
        if name == "mock_feature":
            return MockFeatureClass
        return None

    def get_all_features(self):
        return {"mock_feature": self._feature}



def test_feature_cli_group_registers_expected_commands():
    manager = MockFeatureManager()
    cli_group = FeatureCommandGroup(manager).create_cli_group()

    assert "list" in cli_group.commands
    assert "enable" in cli_group.commands
    assert "disable" in cli_group.commands
    assert "status" in cli_group.commands
    assert "info" in cli_group.commands


def test_feature_list_command_executes_successfully():
    manager = MockFeatureManager()
    cli_group = FeatureCommandGroup(manager).create_cli_group()
    runner = CliRunner()

    result = runner.invoke(cli_group, ["list"])

    assert result.exit_code == 0
    assert "mock_feature" in result.output


def test_feature_info_command_shows_documentation():
    manager = MockFeatureManager()
    cli_group = FeatureCommandGroup(manager).create_cli_group()
    runner = CliRunner()

    result = runner.invoke(cli_group, ["info", "mock_feature"])

    assert result.exit_code == 0
    assert "Mock feature used for testing" in result.output


def test_feature_cli_help_and_invalid_command_handling():
    manager = MockFeatureManager()
    cli_group = FeatureCommandGroup(manager).create_cli_group()
    runner = CliRunner()

    help_result = runner.invoke(cli_group, ["--help"])
    assert help_result.exit_code == 0
    assert "Manage Xencode features" in help_result.output

    invalid_result = runner.invoke(cli_group, ["does-not-exist"])
    assert invalid_result.exit_code != 0
    assert "No such command" in invalid_result.output
