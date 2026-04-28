"""Detect drift between the curated MODEL_NAME_TO_ABBREV registry and what each
provider's models.list() endpoint actually exposes.

Designed to be invoked once per day (gated by ~/.gpt_cli/last_model_check.txt)
right before a query, so that a red warning fires when a provider ships a new
flagship or deprecates an old model. xAI is intentionally excluded -- the
registry hardcodes its single supported model.

The filtering logic mirrors /home/sfish/Dropbox/projects/coherence/utils/measure_529_rate.py
so we get the same "chat-capable, dated, non-experimental" subset the user already
trusts in another project.
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import anthropic
import colorama
from google import genai
from openai import OpenAI

# ── Filter rules (copied from measure_529_rate.py) ──────────────────────────

_ANTHROPIC_EXCLUDED_IDS: set[str] = {
    "claude-3-sonnet-20240229",  # 404
    "claude-3-5-sonnet-20240620",  # 404
    "claude-3-5-sonnet-20241022",  # 529
}

_OPENAI_EXCLUDED_SUBSTRINGS: tuple[str, ...] = (
    "sora",  # video generation
    "deep-research",  # agentic research, not chat
    "audio",  # speech/audio models
    "image",  # image generation/vision-only
    "realtime",  # streaming voice
    "transcribe",  # ASR
    "tts",  # text-to-speech
    "computer-use",  # computer-use agent
    "search",  # web-search augmented
    "moderation",  # content moderation
    "instruct",  # legacy completion-only
    "-pro-",  # only on v1/responses
)

_GOOGLE_EXCLUDED_SUBSTRINGS: tuple[str, ...] = (
    "audio",
    "image",
    "tts",
    "computer-use",
    "robotics",
    "customtools",
    "embedding",
    "latest",  # unversioned aliases
    "-001",  # snapshots no longer available to new users
)

# Registry entries the filter would otherwise mark as drift.
# gpt-4-base lacks a -YYYY suffix, so the OpenAI filter excludes it -- but
# we keep it in the registry for legacy-completions support.
_OPENAI_REGISTRY_ALLOWLIST: set[str] = {"gpt-4-base"}


def _is_excluded_openai(model_id: str) -> bool:
    if any(s in model_id for s in _OPENAI_EXCLUDED_SUBSTRINGS):
        return True
    # Require a 4-digit date component (e.g. -2024, -0125) to filter unversioned aliases
    if not re.search(r"-\d{4}", model_id):
        return True
    return False


def _is_excluded_google(model_id: str) -> bool:
    if not model_id.startswith("gemini-"):
        return True  # excludes gemma, deep-research-pro, nano-banana-pro, etc.
    return any(s in model_id for s in _GOOGLE_EXCLUDED_SUBSTRINGS)


# ── Provider fetchers ──────────────────────────────────────────────────────


def fetch_anthropic_models() -> set[str]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY_CLI"))
    page = client.models.list(limit=1000)
    return {
        m.id
        for m in page.data
        if "krait" not in m.id
        and "fast" not in m.id
        and m.id not in _ANTHROPIC_EXCLUDED_IDS
    }


def fetch_openai_models() -> set[str]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_CLI"))
    return {m.id for m in client.models.list().data if not _is_excluded_openai(m.id)}


def fetch_google_models() -> set[str]:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_CLI"))
    discovered: set[str] = set()
    for m in client.models.list():
        if m.name is None:
            continue
        model_id = m.name.removeprefix("models/")
        if not _is_excluded_google(model_id):
            discovered.add(model_id)
    return discovered


# ── Daily-check marker ─────────────────────────────────────────────────────

_MARKER_PATH = Path("~/.gpt_cli/last_model_check.txt").expanduser()


def should_run_discovery() -> bool:
    """True iff today's date does not match the marker file's contents."""
    if not _MARKER_PATH.exists():
        return True
    return _MARKER_PATH.read_text().strip() != date.today().isoformat()


def mark_check_done() -> None:
    """Stamp today's date into the marker file. Always called after discovery,
    even on failure, so a transient outage doesn't trigger 3 retries per query
    for the rest of the day."""
    _MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MARKER_PATH.write_text(date.today().isoformat())


# ── Comparison / formatting ────────────────────────────────────────────────


def compare_to_registry(
    discovered_anthropic: set[str],
    discovered_openai: set[str],
    discovered_google: set[str],
) -> dict[str, dict[str, set[str]]]:
    """Return {provider: {"missing_from_registry": {...}, "missing_from_api": {...}}}.

    Only providers with non-empty drift are included. The OpenAI allowlist
    (gpt-4-base) is added to the discovered set before diffing so it never
    surfaces as drift despite being filtered out by _is_excluded_openai.
    """
    # Late import to avoid circularity on cold start
    from model_handling import (
        ANTHROPIC_MODEL_TO_ABBREV,
        GOOGLE_MODEL_TO_ABBREV,
        OPENAI_MODEL_TO_ABBREV,
    )

    discovered_openai = discovered_openai | _OPENAI_REGISTRY_ALLOWLIST
    drift: dict[str, dict[str, set[str]]] = {}
    for name, registry, discovered in [
        ("anthropic", set(ANTHROPIC_MODEL_TO_ABBREV), discovered_anthropic),
        ("openai", set(OPENAI_MODEL_TO_ABBREV), discovered_openai),
        ("google", set(GOOGLE_MODEL_TO_ABBREV), discovered_google),
    ]:
        missing_from_registry = discovered - registry
        missing_from_api = registry - discovered
        if missing_from_registry or missing_from_api:
            drift[name] = {
                "missing_from_registry": missing_from_registry,
                "missing_from_api": missing_from_api,
            }
    return drift


def format_drift_warning(drift: dict[str, dict[str, set[str]]]) -> str:
    lines = [f"{colorama.Fore.RED}[model drift detected]"]
    for provider, d in drift.items():
        if d["missing_from_registry"]:
            lines.append(
                f"  {provider}: in API but not registry: {sorted(d['missing_from_registry'])}"
            )
        if d["missing_from_api"]:
            lines.append(
                f"  {provider}: in registry but not API: {sorted(d['missing_from_api'])}"
            )
    lines.append(colorama.Style.RESET_ALL)
    return "\n".join(lines)
