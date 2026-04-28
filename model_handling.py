from typing import Literal


OPENAI_MODEL_TO_ABBREV: dict[str, list[str]] = {
    # GPT-5 family
    "gpt-5.5-2026-04-23": ["5.5"],
    "gpt-5.4-2026-03-05": [],
    "gpt-5.4-mini-2026-03-17": [],
    "gpt-5.4-nano-2026-03-17": [],
    "gpt-5.2-2025-12-11": [],
    "gpt-5.1-2025-11-13": [],
    "gpt-5-2025-08-07": ["5", "gpt-5"],
    "gpt-5-mini-2025-08-07": ["5m", "gpt-5-mini"],
    "gpt-5-nano-2025-08-07": ["5n", "gpt-5-nano"],
    # o-series reasoning models
    "o4-mini-2025-04-16": ["o4-mini", "o4m", "o4"],
    "o3-2025-04-16": ["o3"],
    "o3-mini-2025-01-31": [],
    "o1-2024-12-17": [],
    # GPT-4 family
    "gpt-4.1-2025-04-14": ["41", "4.1"],
    "gpt-4.1-mini-2025-04-14": [],
    "gpt-4.1-nano-2025-04-14": [],
    "gpt-4o-2024-11-20": [],
    "gpt-4o-2024-08-06": ["4o", "gpt-4o"],
    "gpt-4o-2024-05-13": [],
    "gpt-4o-mini-2024-07-18": [],
    "gpt-4-turbo-2024-04-09": [],
    "gpt-4-0613": [],
    "gpt-4-base": ["base"],
    # GPT-3.5
    "gpt-3.5-turbo-0125": [],
    "gpt-3.5-turbo-1106": [],
}

ANTHROPIC_MODEL_TO_ABBREV: dict[str, list[str]] = {
    "claude-opus-4-7": ["c", "co"],
    "claude-opus-4-6": ["co6"],
    "claude-opus-4-5-20251101": ["co5"],
    "claude-opus-4-1-20250805": [],
    "claude-opus-4-20250514": [],
    "claude-sonnet-4-6": ["cs"],
    "claude-sonnet-4-5-20250929": ["cs5"],
    "claude-sonnet-4-20250514": [],
    "claude-haiku-4-5-20251001": ["ch"],
}

GOOGLE_MODEL_TO_ABBREV: dict[str, list[str]] = {
    # Gemini 3 family
    "gemini-3.1-pro-preview": ["g3"],
    "gemini-3.1-flash-lite-preview": [],
    "gemini-3.1-flash-live-preview": [],
    "gemini-3-pro-preview": [],
    "gemini-3-flash-preview": [],
    # Gemini 2.5
    "gemini-2.5-pro": ["g", "gp"],
    "gemini-2.5-flash": ["gf"],
    "gemini-2.5-flash-lite": [],
    # Gemini 2.0
    "gemini-2.0-flash": [],
    "gemini-2.0-flash-lite": [],
}

XAI_MODEL_TO_ABBREV: dict[str, list[str]] = {
    "grok-4-0709": ["x", "grok"],
}

MODEL_NAME_TO_ABBREV: dict[str, list[str]] = {
    **OPENAI_MODEL_TO_ABBREV,
    **ANTHROPIC_MODEL_TO_ABBREV,
    **GOOGLE_MODEL_TO_ABBREV,
    **XAI_MODEL_TO_ABBREV,
}

DEFAULT_MODEL_NAME = "claude-opus-4-7"


def lacks_streaming_support(model_name: str) -> bool:
    assert not uses_legacy_completions(
        model_name
    ), f"Shouldn't be checking this for legacy model {model_name} -- shouldn't be using chat completions API"
    return False


def is_reasoning_model(model_name: str) -> bool:
    return model_name in {"o3-2025-04-16", "o4-mini-2025-04-16"}


_GPT_5_FAMILY: set[str] = {
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5.1-2025-11-13",
    "gpt-5.2-2025-12-11",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5.5-2026-04-23",
}


def supports_reasoning_effort(model_name: str) -> bool:
    """Returns True if model supports reasoning_effort parameter."""
    return model_name in (_GPT_5_FAMILY | {"o3-2025-04-16", "o4-mini-2025-04-16"})


def supports_verbosity(model_name: str) -> bool:
    """Returns True if model supports verbosity parameter."""
    return model_name in _GPT_5_FAMILY


MODEL_NAME_TO_ABBREV_LEGEND = ", ".join(
    [
        f"{short_str_list} for {model_name}"
        for model_name, short_str_list in MODEL_NAME_TO_ABBREV.items()
    ]
)


def extract_model_name(model_short_str: str) -> str:
    """
    Extract the model name from a string input which was inputted via command line.
    """
    for model_name, short_str_list in MODEL_NAME_TO_ABBREV.items():
        if model_short_str in short_str_list or model_short_str == model_name:
            return model_name
    raise NotImplementedError(f"Can't recognize model name {model_short_str}")


def model_name_to_provider(
    model_name: str,
) -> Literal["anthropic", "openai", "google", "xai"]:
    if model_name in ANTHROPIC_MODEL_TO_ABBREV:
        return "anthropic"
    elif model_name in OPENAI_MODEL_TO_ABBREV:
        return "openai"
    elif model_name in GOOGLE_MODEL_TO_ABBREV:
        return "google"
    elif model_name in XAI_MODEL_TO_ABBREV:
        return "xai"
    else:
        raise NotImplementedError(f"unrecognized {model_name}")


def uses_legacy_completions(model_name: str) -> bool:
    return model_name == "gpt-4-base"
