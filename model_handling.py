from typing import Literal


# OpenAI
GPT_41_MODEL_NAME = "gpt-4.1-2025-04-14"
O4_MINI_MODEL_NAME = "o4-mini-2025-04-16"
O3_MODEL_NAME = "o3-2025-04-16"
GPT_4_BASE = "gpt-4-base"

OPENAI_MODELS = [
    GPT_41_MODEL_NAME,
    O3_MODEL_NAME,
    O4_MINI_MODEL_NAME,
    GPT_4_BASE,
]

# Anthropic models
CLAUDE_4_OPUS_MODEL_NAME = "claude-opus-4-20250514"
CLAUDE_4_SONNET_MODEL_NAME = "claude-sonnet-4-20250514"

ANTHROPIC_MODELS = [
    CLAUDE_4_OPUS_MODEL_NAME,
    CLAUDE_4_SONNET_MODEL_NAME,
]

# Google models

GEMINI_2_5_MODEL_NAME = "gemini-2.5-pro-preview-06-05"

GOOGLE_MODELS = [GEMINI_2_5_MODEL_NAME]

GROK_3_MODEL_NAME = "grok-3-fast-beta"
GROK_3_MINI_MODEL_NAME = "grok-3-mini-fast-beta"
GROK_4_MODEL_NAME = "grok-4-0709"
XAI_MODELS = [GROK_3_MODEL_NAME, GROK_3_MINI_MODEL_NAME, GROK_4_MODEL_NAME]

DEFAULT_MODEL_NAME = CLAUDE_4_OPUS_MODEL_NAME


def lacks_streaming_support(model_name: str) -> bool:
    assert not uses_legacy_completions(
        model_name
    ), f"Shouldn't be checking this for legacy model {model_name} -- shouldn't be using chat completions API"
    return False


def is_reasoning_model(model_name: str) -> bool:
    if model_name == O4_MINI_MODEL_NAME or model_name == O3_MODEL_NAME:
        return True
    else:
        return False


MODEL_NAME_TO_ABBREV = {
    GPT_41_MODEL_NAME: ["41", "4.1"],
    O4_MINI_MODEL_NAME: ["o4-mini", "o4m", "o4"],
    O3_MODEL_NAME: ["o3"],
    GPT_4_BASE: ["base"],
    CLAUDE_4_OPUS_MODEL_NAME: ["c"],
    CLAUDE_4_SONNET_MODEL_NAME: ["cs"],
    GEMINI_2_5_MODEL_NAME: ["g"],
    GROK_3_MODEL_NAME: ["x3"],
    GROK_3_MINI_MODEL_NAME: ["xm", "xm3"],
    GROK_4_MODEL_NAME: ["x", "grok"],
}

MODEL_NAME_TO_ABBREV_LEGEND = ", ".join(
    [
        f"{short_str_list} for {model_name}"
        for model_name, short_str_list in MODEL_NAME_TO_ABBREV.items()
    ]
)


def extract_model_name(model_short_str):
    """
    Extract the model name from a string input which was inputted via command line.
    """
    for model_name, short_str_list in MODEL_NAME_TO_ABBREV.items():
        if model_short_str in short_str_list or model_short_str == model_name:
            return model_name
    else:
        raise NotImplementedError(f"Can't recognize model name {model_short_str}")


def model_name_to_provider(
    model_name: str,
) -> Literal["anthropic", "openai", "google", "xai"]:
    if model_name in ANTHROPIC_MODELS:
        return "anthropic"
    elif model_name in OPENAI_MODELS:
        return "openai"
    elif model_name in GOOGLE_MODELS:
        return "google"
    elif model_name in XAI_MODELS:
        return "xai"
    else:
        raise NotImplementedError(f"unrecognized {model_name}")


def uses_legacy_completions(model_name: str) -> bool:
    return model_name in [GPT_4_BASE]
