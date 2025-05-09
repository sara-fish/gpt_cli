# GPT-4 models
from typing import Literal


GPT_4_0613_MODEL_NAME = "gpt-4-0613"

# GPT-4o
GPT_4o_MODEL_NAME = "gpt-4o-2024-11-20"
GPT_4o_SEARCH_MODEL_NAME = "gpt-4o-search-preview-2025-03-11"

# GPT-4-base
GPT_4_BASE_MODEL_NAME = "gpt-4-base"

# GPT-4.5
GPT_45_MODEL_NAME = "gpt-4.5-preview-2025-02-27"

# o1 and o3 series
O3_MINI_MODEL_NAME = "o3-mini-2025-01-31"
O4_MINI_MODEL_NAME = "o4-mini-2025-04-16"
O3_MODEL_NAME = "o3-2025-04-16"

OPENAI_MODELS = [
    GPT_4_0613_MODEL_NAME,
    GPT_4o_MODEL_NAME,
    GPT_4o_SEARCH_MODEL_NAME,
    GPT_45_MODEL_NAME,
    GPT_4_BASE_MODEL_NAME,
    O3_MINI_MODEL_NAME,
    O4_MINI_MODEL_NAME,
    O3_MODEL_NAME,
]

# Anthropic models

CLAUDE_37_SONNET_MODEL_NAME = "claude-3-7-sonnet-20250219"

ANTHROPIC_MODELS = [
    CLAUDE_37_SONNET_MODEL_NAME,
]

# Google models

GEMINI_2_5_MODEL_NAME = "gemini-2.5-pro-preview-05-06"

GOOGLE_MODELS = [GEMINI_2_5_MODEL_NAME]

GROK_3_MODEL_NAME = "grok-3-fast-beta"
GROK_3_MINI_MODEL_NAME = "grok-3-mini-fast-beta"
XAI_MODELS = [GROK_3_MODEL_NAME, GROK_3_MINI_MODEL_NAME]

DEFAULT_MODEL_NAME = CLAUDE_37_SONNET_MODEL_NAME


def uses_legacy_completions(model_name: str) -> bool:
    if model_name == GPT_4_BASE_MODEL_NAME:
        return True
    else:
        return False


def lacks_streaming_support(model_name: str) -> bool:
    if model_name == O3_MODEL_NAME:
        return True
    else:
        return False


def is_reasoning_model(model_name: str) -> bool:
    if (
        model_name == O3_MINI_MODEL_NAME
        or model_name == O4_MINI_MODEL_NAME
        or model_name == O3_MODEL_NAME
    ):
        return True
    else:
        return False


def get_system_name(model_name: str) -> str:
    return "system"


MODEL_NAME_TO_ABBREV = {
    GPT_4_0613_MODEL_NAME: ["4"],
    GPT_4_BASE_MODEL_NAME: ["base"],
    GPT_4o_MODEL_NAME: ["4o"],
    GPT_4o_SEARCH_MODEL_NAME: ["4os"],
    GPT_45_MODEL_NAME: ["45", "4.5"],
    O3_MINI_MODEL_NAME: ["o3-mini", "o3m"],
    O3_MODEL_NAME: ["o3"],
    O4_MINI_MODEL_NAME: ["o4-mini", "o4m"],
    CLAUDE_37_SONNET_MODEL_NAME: ["c"],
    GEMINI_2_5_MODEL_NAME: ["g"],
    GROK_3_MODEL_NAME: ["x", "x3"],
    GROK_3_MINI_MODEL_NAME: ["xm", "xm3"],
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


def model_name_to_provider(model_name: str) -> Literal["anthropic", "openai", "google"]:
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
