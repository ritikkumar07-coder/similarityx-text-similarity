"""Streamlit application smoke tests."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_starts_on_text_comparison():
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)

    assert not app.exception
    assert app.sidebar.radio
    assert app.text_area


def test_empty_comparison_shows_validation_warning():
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)

    compare_button = next(button for button in app.button if button.label == "Compare text")
    compare_button.click()
    app.run(timeout=30)

    assert app.warning


def test_example_button_populates_and_clear_button_resets_inputs():
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)

    example_button = next(button for button in app.button if button.label == "Example 1")
    example_button.click()
    app.run(timeout=30)

    assert len(app.text_area) == 2
    assert app.text_area[0].value
    assert app.text_area[1].value

    clear_button = next(button for button in app.button if button.label == "Clear all")
    clear_button.click()
    app.run(timeout=30)

    assert app.text_area[0].value == ""
    assert app.text_area[1].value == ""


def test_sidebar_navigation_reaches_all_application_pages():
    app = AppTest.from_file(APP_PATH)
    app.run(timeout=30)

    app.sidebar.radio[0].set_value("Methodology")
    app.run(timeout=30)
    assert not app.exception

    app.sidebar.radio[0].set_value("About Project")
    app.run(timeout=30)
    assert not app.exception
