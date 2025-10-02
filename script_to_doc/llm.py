"""Utilities for integrating OpenAI models into the transcript pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

from openai import OpenAI


LOGGER = logging.getLogger(__name__)


class OpenAIConfigurationError(RuntimeError):
    """Raised when the OpenAI client cannot be initialised."""


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIConfigurationError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def _parse_json_content(text: str) -> dict:
    """Attempt to parse JSON while tolerating markdown fences or extra text."""
    text = text.strip()

    # Remove code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```json?", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            return json.loads(candidate)
        raise


def _usage_to_dict(usage) -> Dict[str, int]:
    if not usage:
        return {}
    if hasattr(usage, "model_dump"):
        return {k: v for k, v in usage.model_dump().items() if isinstance(v, (int, float))}
    if hasattr(usage, "to_dict"):
        return {k: v for k, v in usage.to_dict().items() if isinstance(v, (int, float))}
    if isinstance(usage, dict):
        return {k: v for k, v in usage.items() if isinstance(v, (int, float))}
    result = {}
    for key in dir(usage):
        if key.startswith("_"):
            continue
        value = getattr(usage, key, None)
        if isinstance(value, (int, float)):
            result[key] = value
    return result


def summarize_with_openai(
    transcript: str,
    *,
    model: str,
    max_sentences: int,
    temperature: float = 0.2,
    tone: str = "Neutral",
    audience: str = "General",
) -> Tuple[List[str], Dict[str, int]]:
    """Use an OpenAI model to identify key sentences in the transcript."""
    client = _get_client()
    prompt = (
        "You are helping transform meeting transcripts into concise training documents. "
        "Extract the most important actionable sentences that should be included in the summary. "
        f"Write in a {tone.lower()} tone for {audience}. "
        "Respond with JSON containing a 'summary_sentences' array of strings."
    )
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": prompt + " Always respond with pure JSON."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Transcript:\n" + transcript + "\n\nReturn JSON exactly shaped as {\"summary_sentences\": [string, ...]}"
                        ),
                    }
                ],
            },
        ],
    )
    message = response.output[0].content[0].text
    LOGGER.debug("OpenAI raw summary response: %s", message)
    data = _parse_json_content(message)
    sentences = data.get("summary_sentences", [])
    if not isinstance(sentences, list):
        raise ValueError("OpenAI summary response did not contain a list in 'summary_sentences'")
    usage = _usage_to_dict(getattr(response, "usage", None))
    return [str(sentence) for sentence in sentences][:max_sentences], usage


def expand_steps_with_openai(
    steps: Sequence[str],
    *,
    model: str,
    temperature: float = 0.3,
    tone: str = "Neutral",
    audience: str = "General",
) -> Tuple[List[str], Dict[str, int]]:
    """Request expanded training guidance for each step from an OpenAI model."""
    client = _get_client()
    prompt = (
        "Expand each procedural step into a detailed training paragraph. "
        "Reference best practices, required tools, and success criteria. Return JSON with an 'expansions' array"
        " of objects like {\"step\": string, \"details\": string}. "
        f"Write in a {tone.lower()} tone tailored for {audience}."
    )
    steps_payload = "\n".join(f"Step {idx}: {text}" for idx, text in enumerate(steps, start=1))
    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {
                "role": "system",
                "content": (
                    prompt
                    + " Always respond with JSON formatted as {\"expansions\": [{\"step\": string, \"details\": string}]}."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Here are the steps that require expansion:\n"
                            f"{steps_payload}\n\nReturn only the JSON object."
                        ),
                    }
                ],
            },
        ],
    )
    message = response.output[0].content[0].text
    data = _parse_json_content(message)
    expansions: Iterable[dict] = data.get("expansions", [])
    expanded: List[str] = []
    for idx, item in enumerate(expansions, start=1):
        if isinstance(item, dict) and "details" in item:
            expanded.append(str(item["details"]))
    # Ensure list is aligned with original steps length.
    while len(expanded) < len(steps):
        expanded.append("Provide detailed context, expected outcomes, and key references for this step.")
    usage = _usage_to_dict(getattr(response, "usage", None))
    return expanded[: len(steps)], usage


def generate_training_plan_with_openai(
    transcript: str,
    *,
    model: str,
    temperature: float = 0.25,
    max_steps: int = 12,
    tone: str = "Neutral",
    audience: str = "General",
) -> Tuple[List[dict], Dict[str, int]]:
    """Ask an OpenAI model to extract procedural steps and training guidance."""
    client = _get_client()
    system_prompt = (
        "You convert meeting transcripts into comprehensive training guides. "
        "Infer the underlying process even if the dialogue contains greetings, questions, or fragmented phrases. "
        "Always produce at least three sequential training steps. "
        "Each step must include a clear title, a concise summary, an in-depth explanation, and 2-4 actionable bullet points. "
        f"Write the document in a {tone.lower()} tone tailored for {audience}. "
        "Return only valid JSON."
    )
    user_prompt = (
        "Meeting transcript (cleaned):\n"
        f"{transcript}\n\nReturn JSON exactly matching the schema: {{\"steps\": [{{\"title\": string, \"summary\": string, \"details\": string, \"actions\": [string, ...]}}]}}."
    )
    LOGGER.debug(
        "Submitting training-plan prompt to OpenAI",
        extra={"model": model, "chars": len(transcript)},
    )

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    message = response.output[0].content[0].text
    LOGGER.debug("OpenAI raw plan response: %s", message)

    try:
        data = _parse_json_content(message)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to decode OpenAI plan JSON: %s", exc)
        raise
    items = data.get("steps", [])
    if not isinstance(items, list):  # pragma: no cover - defensive
        raise ValueError("OpenAI training plan response missing 'steps' list")

    plan: List[dict] = []
    for item in items[:max_steps]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Process Step").strip()
        summary = str(item.get("summary") or "").strip()
        details = str(item.get("details") or item.get("instructions") or "").strip()
        actions_raw = item.get("actions") or []
        actions = [str(action).strip() for action in actions_raw if str(action).strip()]
        plan.append(
            {
                "title": title,
                "summary": summary,
                "details": details,
                "actions": actions,
            }
        )
    if not plan:
        LOGGER.warning("OpenAI returned an empty training plan")
    usage = _usage_to_dict(getattr(response, "usage", None))
    return plan, usage
