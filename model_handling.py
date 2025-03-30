# GPT-4 models
from typing import Literal


GPT_4_0314_MODEL_NAME = "gpt-4-0314"
GPT_4_0613_MODEL_NAME = "gpt-4-0613"
GPT_4_MODEL_NAME = "gpt-4"

# GPT-4-32k
GPT_4_32K_0314_MODEL_NAME = "gpt-4-32k-0314"
GPT_4_32K_0613_MODEL_NAME = "gpt-4-32k-0613"
GPT_4_32K_MODEL_NAME = "gpt-4-32k"

# GPT-4o
GPT_4o_MODEL_NAME = "gpt-4o-2024-11-20"
GPT_4o_SEARCH_MODEL_NAME = "gpt-4o-search-preview-2025-03-11"

# GPT-4-base
GPT_4_BASE_MODEL_NAME = "gpt-4-base"

# GPT-4.5
GPT_45_MODEL_NAME = "gpt-4.5-preview-2025-02-27"

# o1 and o3 series
O1_MODEL_NAME = "o1-2024-12-17"
O1_MINI_MODEL_NAME = "o1-mini-2024-09-12"
O3_MINI_MODEL_NAME = "o3-mini-2025-01-31"

OPENAI_MODELS = [
    GPT_4_0314_MODEL_NAME,
    GPT_4_0613_MODEL_NAME,
    GPT_4_MODEL_NAME,
    GPT_4_32K_0314_MODEL_NAME,
    GPT_4_32K_0613_MODEL_NAME,
    GPT_4_32K_MODEL_NAME,
    GPT_4o_MODEL_NAME,
    GPT_4o_SEARCH_MODEL_NAME,
    GPT_45_MODEL_NAME,
    GPT_4_BASE_MODEL_NAME,
    O1_MODEL_NAME,
    O1_MINI_MODEL_NAME,
    O3_MINI_MODEL_NAME,
]

# Anthropic models

CLAUDE_35_SONNET_MODEL_NAME = "claude-3-5-sonnet-20241022"
CLAUDE_37_SONNET_MODEL_NAME = "claude-3-7-sonnet-20250219"

ANTHROPIC_MODELS = [
    CLAUDE_37_SONNET_MODEL_NAME,
    CLAUDE_35_SONNET_MODEL_NAME,
]

# Google models

GEMINI_15_PRO_MODEL_NAME = "gemini-1.5-pro-002"
GEMINI_2_MODEL_NAME = "gemini-2.0-flash-exp"

GOOGLE_MODELS = [GEMINI_15_PRO_MODEL_NAME, GEMINI_2_MODEL_NAME]


def uses_legacy_completions(model_name: str) -> bool:
    if model_name == GPT_4_BASE_MODEL_NAME:
        return True
    else:
        return False


def lacks_streaming_support(model_name: str) -> bool:
    if model_name == O1_MODEL_NAME or model_name == O1_MINI_MODEL_NAME:
        return True
    else:
        return False


def is_reasoning_model(model_name: str) -> bool:
    if (
        model_name == O1_MODEL_NAME
        or model_name == O1_MINI_MODEL_NAME
        or model_name == O3_MINI_MODEL_NAME
    ):
        return True
    else:
        return False


def get_system_name(model_name: str) -> str:
    if model_name == O1_MINI_MODEL_NAME:
        return "user"
    elif model_name == O1_MODEL_NAME:
        return "developer"
    else:
        return "system"


MODEL_NAME_TO_ABBREV = {
    GPT_4_0314_MODEL_NAME: ["4-0314", "4-old", "4"],
    GPT_4_0613_MODEL_NAME: ["4-0613"],
    GPT_4_MODEL_NAME: ["4-new"],
    GPT_4_32K_MODEL_NAME: ["32k", "4-32k"],
    GPT_4_32K_0314_MODEL_NAME: ["32k-0314"],
    GPT_4_32K_0613_MODEL_NAME: ["32k-0613"],
    GPT_4_BASE_MODEL_NAME: ["base"],
    GPT_4o_MODEL_NAME: ["4o"],
    GPT_4o_SEARCH_MODEL_NAME: ["4os"],
    GPT_45_MODEL_NAME: ["45", "4.5"],
    O1_MODEL_NAME: ["o1", "1"],
    O1_MINI_MODEL_NAME: ["o1-mini", "o1m"],
    O3_MINI_MODEL_NAME: ["o3-mini", "o3", "o3m", "3"],
    CLAUDE_35_SONNET_MODEL_NAME: ["claude3.5-sonnet", "c3.5", "c35"],
    CLAUDE_37_SONNET_MODEL_NAME: ["claude3.7-sonnet", "c3.7", "c"],
    GEMINI_15_PRO_MODEL_NAME: ["g1", "g15", "gemini1.5-pro"],
    GEMINI_2_MODEL_NAME: ["g", "g2"],
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
    else:
        raise NotImplementedError(f"unrecognized {model_name}")
