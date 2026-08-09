import pytest

from resume_crew.pipeline import split_writer_output


def test_splits_expected_sections() -> None:
    bullets, guide = split_writer_output(
        "## Tailored Resume Bullets\n- A\n\n## Interview Prep Guide\n1. Q"
    )
    assert bullets == "- A"
    assert guide == "1. Q"


def test_rejects_malformed_writer_output() -> None:
    with pytest.raises(ValueError, match="invalid"):
        split_writer_output("Some advice")


def test_word_export_is_not_part_of_the_pipeline_module() -> None:
    import resume_crew.pipeline as pipeline

    assert not hasattr(pipeline, "export_docx")


def test_splits_with_crlf_line_endings() -> None:
    """BUG-05: CRLF line endings from some LLMs must not break section parsing."""
    text = "## Tailored Resume Bullets\r\n- Bullet one\r\n\r\n## Interview Prep Guide\r\n1. Question"
    bullets, guide = split_writer_output(text)
    assert "Bullet one" in bullets
    assert "Question" in guide


def test_splits_with_level_three_headings() -> None:
    """BUG-05: ### headings (level 3) must be accepted, not just ##."""
    text = "### Tailored Resume Bullets\n- B\n\n### Interview Prep Guide\n1. Q"
    bullets, guide = split_writer_output(text)
    assert bullets == "- B"
    assert guide == "1. Q"


def test_splits_with_spaced_heading_words() -> None:
    """BUG-05: extra whitespace inside heading text should still match."""
    text = "##  Tailored  Resume  Bullets \n- B\n\n##  Interview  Prep  Guide \n1. Q"
    bullets, guide = split_writer_output(text)
    assert "B" in bullets
    assert "Q" in guide
