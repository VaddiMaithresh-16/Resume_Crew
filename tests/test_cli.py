import os

from resume_crew.cli import _disable_external_tracing, build_parser


def test_cli_has_no_word_export_flag() -> None:
    parser = build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--export-docx" not in option_strings


def test_cli_forces_tracing_and_telemetry_off() -> None:
    _disable_external_tracing()
    assert os.environ["CREWAI_TRACING_ENABLED"] == "false"
    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true"
    assert os.environ["CREWAI_DISABLE_TRACKING"] == "true"
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def test_cli_has_no_history_flag() -> None:
    parser = build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--history" not in option_strings


def test_cli_has_no_print_ollama_env_flag() -> None:
    parser = build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--print-ollama-env" not in option_strings


def test_cli_has_no_output_flag() -> None:
    """The custom --output directory feature was removed; runs always auto-name their folder."""
    parser = build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--output" not in option_strings
