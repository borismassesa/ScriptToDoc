import tempfile
from pathlib import Path

import pytest

from script_to_doc.pipeline import PipelineConfig, process_transcript


class FakeToken:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos_ = "VERB" if text.lower() in {"log", "configure", "create"} else "NOUN"


class FakeDoc(list):
    def __init__(self, words):
        super().__init__(FakeToken(word.strip(".,")) for word in words if word)


def fake_nlp(text: str):
    return FakeDoc(text.split())


def fake_sent_tokenize(text: str):
    return [sentence.strip() for sentence in text.split(".") if sentence.strip()]


def fake_generate_plan(*args, **kwargs):
    raise RuntimeError("LLM unavailable in test")


@pytest.fixture(autouse=True)
def _no_nltk(monkeypatch):
    monkeypatch.setattr("script_to_doc.pipeline.ensure_nltk_resources", lambda: None)
    monkeypatch.setattr("script_to_doc.pipeline.sent_tokenize", fake_sent_tokenize)


def test_process_transcript_falls_back_when_llm_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "script_to_doc.pipeline.generate_training_plan_with_openai",
        fake_generate_plan,
    )

    transcript_path = tmp_path / "meeting.txt"
    transcript_path.write_text("Log in to the tool. Configure the dashboard. Create tasks for the team.")
    output_dir = tmp_path / "out"

    config = PipelineConfig(
        use_llm_workflow=True,
        use_llm_summary=False,
        use_llm_expander=False,
    )

    result = process_transcript(transcript_path, output_dir, config=config, nlp=fake_nlp)

    assert result["steps"]  # fallback extracted steps
    assert result["plan"]  # structured plan exists for preview
    metrics = result.get("metrics", {})
    assert "duration_sec" in metrics
    assert metrics.get("tokens", {}) == {}
*** End Patch
