"""Integration tests for core TUI feature navigation components."""

import pytest
from textual.widgets import Button

from xencode.tui.features.base_feature_panel import BaseFeaturePanel, FeatureStatus
from xencode.tui.features.feature_navigator import FeatureNavigator


class DummyPanel(BaseFeaturePanel):
    """Concrete panel for testing base panel behavior."""

    def __init__(self):
        super().__init__(feature_name="dummy", title="Dummy Feature")



def test_feature_status_indicator_renders_and_updates_class():
    indicator = FeatureStatus("dummy")

    indicator.status = "enabled"
    indicator.watch_status("enabled")

    assert "DUMMY" in indicator.render()
    assert "enabled" in indicator.classes


@pytest.mark.asyncio
async def test_feature_navigator_updates_selected_feature():
    navigator = FeatureNavigator()
    button = Button("Code Review", id="nav-code_review")
    event = Button.Pressed(button)

    await navigator.on_button_pressed(event)

    assert navigator.current_feature == "code_review"


@pytest.mark.asyncio
async def test_feature_navigator_ignores_non_feature_buttons():
    navigator = FeatureNavigator()
    button = Button("Other", id="other")
    event = Button.Pressed(button)

    await navigator.on_button_pressed(event)

    assert navigator.current_feature is None


def test_base_feature_panel_set_status_updates_indicator():
    panel = DummyPanel()
    panel.status_indicator = FeatureStatus("dummy")

    panel.set_status("loading")

    assert panel.status_indicator.status == "loading"
