"""
The one place that talks to a language model.

Everything else in the project calls `complete()` and knows nothing about which
provider is behind it. Swapping Gemini for Groq, Ollama or anything else means
editing this file only.

Why the fallback chain
----------------------
Free-tier quota on Google AI Studio is per-model, not per-account, and probing
this key found three of seven candidate models already returning 429
RESOURCE_EXHAUSTED before we had made a single real request. An agent issues
several calls per user question, so hitting a limit mid-demo is a question of
when, not if. `complete()` therefore walks a list of known-working models and
moves to the next one on quota or availability errors.

If every model fails it raises LLMUnavailable rather than returning nonsense.
Callers are expected to catch it and fall back to the rule-based router -- the
analytics and the two ML models work perfectly well without any LLM, and the
demo must not die because a free quota reset.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve .env from the project root, not the current working directory.
# python-dotenv's default search starts at the *calling script's* folder, which
# silently returns no key when a script runs from somewhere else.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Ordered by measured round-trip time, fastest first -- not by model prestige.
# Timing the chain showed gemini-flash-latest and gemini-flash-lite-latest were
# already returning 429 on this key, and because they sat at the front, every
# call burned a second or more failing against them before reaching a model that
# actually answers. The lite models are both quicker and less contended; the
# work here is picking one of four tool names and summarising numbers pandas
# already computed, which does not need a frontier model.
MODEL_CHAIN = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

# The model that answered last. Quota state changes during a session, so rather
# than rediscovering it on every call we try the known-good one first. Module
# level, so it survives Streamlit's reruns within one process.
_PREFERRED: str | None = None


def _ordered_models() -> list[str]:
    if _PREFERRED and _PREFERRED in MODEL_CHAIN:
        return [_PREFERRED] + [m for m in MODEL_CHAIN if m != _PREFERRED]
    return list(MODEL_CHAIN)

# Errors that mean "try a different model" rather than "the request was wrong".
_RETRYABLE = ("RESOURCE_EXHAUSTED", "429", "NOT_FOUND", "404",
              "UNAVAILABLE", "503", "INTERNAL", "500")


class LLMUnavailable(RuntimeError):
    """No model in the chain could answer. Callers should degrade gracefully."""


def api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY")


def is_available() -> bool:
    """Cheap check -- does a key exist at all? Does not make a network call."""
    return bool(api_key())


def _client():
    from google import genai
    key = api_key()
    if not key:
        raise LLMUnavailable(
            "GOOGLE_API_KEY not found. Create a .env file in the project root "
            "containing:  GOOGLE_API_KEY=your-key-here"
        )
    return genai.Client(api_key=key)


def _config(types, system_instruction, temperature, max_output_tokens,
            json_mode, no_thinking):
    kwargs = dict(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        response_mime_type="application/json" if json_mode else None,
    )
    if no_thinking:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def complete(prompt: str,
             system_instruction: str | None = None,
             temperature: float = 0.2,
             max_output_tokens: int = 1200,
             json_mode: bool = False) -> str:
    """
    Send a prompt, get text back. Tries each model in MODEL_CHAIN in turn.

    temperature defaults low: this agent reports numbers computed by pandas, and
    creative rephrasing of a dollar figure is a bug, not a feature.

    json_mode asks the API itself to guarantee valid JSON. Without it, models
    wrap their JSON in ```json fences perhaps a third of the time, and every
    project ends up writing the same brittle string-stripping code. Let the API
    enforce it instead.

    Thinking is switched off, which is not an optimisation -- it is a bug fix.
    Recent Gemini models spend `max_output_tokens` on hidden reasoning *before*
    emitting any visible text, so a generous-looking 800-token budget returned
    answers chopped off mid-sentence and JSON that would not parse. This agent
    needs a tool name and a two-line summary of numbers pandas already computed;
    there is nothing here worth reasoning about. Older models reject the option,
    so a rejection retries the same model without it.
    """
    global _PREFERRED
    from google.genai import types

    client = _client()
    errors: list[str] = []

    for model in _ordered_models():
        for no_thinking in (True, False):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt,
                    config=_config(types, system_instruction, temperature,
                                   max_output_tokens, json_mode, no_thinking),
                )
                text = (response.text or "").strip()
                if text:
                    _PREFERRED = model      # try this one first next time
                    return text
                errors.append(f"{model}: empty response")
                break
            except Exception as exc:                  # noqa: BLE001
                message = str(exc)
                errors.append(f"{model}: {type(exc).__name__}")
                if no_thinking and "INVALID_ARGUMENT" in message:
                    continue      # model dislikes thinking_config -- retry without
                if any(token in message for token in _RETRYABLE):
                    break         # quota or availability -- try the next model
                raise             # a genuine bug in our request; do not mask it

    raise LLMUnavailable("All models failed -> " + "; ".join(errors))


def complete_stream(prompt: str,
                    system_instruction: str | None = None,
                    temperature: float = 0.2,
                    max_output_tokens: int = 1200):
    """
    Same as complete(), but yields the answer in pieces as it is generated.

    The total time is unchanged -- the win is entirely in what it feels like.
    Waiting 1.5 seconds for a paragraph to appear all at once reads as slow;
    seeing the first words after 300ms reads as immediate, even though the last
    word arrives at exactly the same moment.
    """
    global _PREFERRED
    from google.genai import types

    client = _client()
    errors: list[str] = []

    for model in _ordered_models():
        for no_thinking in (True, False):
            try:
                stream = client.models.generate_content_stream(
                    model=model, contents=prompt,
                    config=_config(types, system_instruction, temperature,
                                   max_output_tokens, False, no_thinking),
                )
                produced = False
                for chunk in stream:
                    piece = chunk.text or ""
                    if piece:
                        produced = True
                        yield piece
                if produced:
                    _PREFERRED = model
                    return
                errors.append(f"{model}: empty response")
                break
            except Exception as exc:                  # noqa: BLE001
                message = str(exc)
                errors.append(f"{model}: {type(exc).__name__}")
                if no_thinking and "INVALID_ARGUMENT" in message:
                    continue
                if any(token in message for token in _RETRYABLE):
                    break
                raise

    raise LLMUnavailable("All models failed -> " + "; ".join(errors))


def which_model_works() -> str | None:
    """Diagnostic helper: returns the first model that actually answers."""
    for model in MODEL_CHAIN:
        try:
            complete("Reply with: OK", max_output_tokens=10)
            return model
        except LLMUnavailable:
            return None
        except Exception:                             # noqa: BLE001
            continue
    return None


if __name__ == "__main__":
    print("API key found :", is_available())
    try:
        print("Response      :", complete("Reply with exactly: WORKING"))
        print("Status        : LLM layer ready")
    except LLMUnavailable as exc:
        print("Status        : unavailable ->", exc)
