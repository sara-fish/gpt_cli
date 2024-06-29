# GPT-4 models
from typing import Literal


GPT_4_0314_MODEL_NAME = "gpt-4-0314"
GPT_4_0613_MODEL_NAME = "gpt-4-0613"
GPT_4_MODEL_NAME = "gpt-4"

# GPT-4-turbo
GPT_4_TURBO_1106_MODEL_NAME = "gpt-4-1106-preview"
GPT_4_TURBO_0125_MODEL_NAME = "gpt-4-0125-preview"
GPT_4_TURBO_MODEL_NAME = "gpt-4-turbo-preview"

# GPT-4o
GPT_4o_MODEL_NAME = "gpt-4o-2024-05-13"

# GPT-3.5-turbo
GPT_35_TURBO_0613_MODEL_NAME = "gpt-3.5-turbo-0613"
GPT_35_TURBO_0125_MODEL_NAME = "gpt-3.5-turbo-0125"
GPT_35_MODEL_NAME = "gpt-3.5-turbo-0613"

# GPT-4-32k
GPT_4_32K_0314_MODEL_NAME = "gpt-4-32k-0314"
GPT_4_32K_0613_MODEL_NAME = "gpt-4-32k-0613"
GPT_4_32K_MODEL_NAME = "gpt-4-32k"

# GPT-4-base
GPT_4_BASE_MODEL_NAME = "gpt-4-base"

OPENAI_MODELS = [
    GPT_4_0314_MODEL_NAME,
    GPT_4_0613_MODEL_NAME,
    GPT_4_MODEL_NAME,
    GPT_4_TURBO_1106_MODEL_NAME,
    GPT_4_TURBO_0125_MODEL_NAME,
    GPT_4_TURBO_MODEL_NAME,
    GPT_35_TURBO_0613_MODEL_NAME,
    GPT_35_TURBO_0125_MODEL_NAME,
    GPT_35_MODEL_NAME,
    GPT_4_32K_0314_MODEL_NAME,
    GPT_4_32K_0613_MODEL_NAME,
    GPT_4_32K_MODEL_NAME,
    GPT_4_BASE_MODEL_NAME,
    GPT_4o_MODEL_NAME,
]

# Anthropic models
CLAUDE_3_OPUS_MODEL_NAME = "claude-3-opus-20240229"
CLAUDE_3_SONNET_MODEL_NAME = "claude-3-sonnet-20240229"
CLAUDE_3_HAIKU_MODEL_NAME = "claude-3-haiku-20240307"

CLAUDE_35_SONNET_MODEL_NAME = "claude-3-5-sonnet-20240620"

ANTHROPIC_MODELS = [
    CLAUDE_3_OPUS_MODEL_NAME,
    CLAUDE_3_SONNET_MODEL_NAME,
    CLAUDE_3_HAIKU_MODEL_NAME,
    CLAUDE_35_SONNET_MODEL_NAME,
]


def uses_legacy_completions(model_name: str) -> bool:
    if model_name == GPT_4_BASE_MODEL_NAME:
        return True
    else:
        return False


MODEL_NAME_TO_ABBREV = {
    GPT_4_0314_MODEL_NAME: ["4-0314", "4-old", "4"],
    GPT_4_0613_MODEL_NAME: ["4-0613"],
    GPT_4_MODEL_NAME: ["4-new"],
    GPT_4_TURBO_1106_MODEL_NAME: ["4t-1106"],
    GPT_4_TURBO_0125_MODEL_NAME: ["4t-0125"],
    GPT_4_TURBO_MODEL_NAME: ["4t", "turbo"],
    GPT_35_TURBO_0613_MODEL_NAME: ["3-0613"],
    GPT_35_TURBO_0125_MODEL_NAME: ["3-0125"],
    GPT_35_MODEL_NAME: ["3"],
    GPT_4_32K_MODEL_NAME: ["32k", "4-32k"],
    GPT_4_32K_0314_MODEL_NAME: ["32k-0314"],
    GPT_4_32K_0613_MODEL_NAME: ["32k-0613"],
    GPT_4_BASE_MODEL_NAME: ["base"],
    GPT_4o_MODEL_NAME: ["4o"],
    CLAUDE_3_OPUS_MODEL_NAME: ["claude3", "co"],
    CLAUDE_3_SONNET_MODEL_NAME: ["claude3-sonnet", "cs"],
    CLAUDE_3_HAIKU_MODEL_NAME: ["claude3-haiku", "ch"],
    CLAUDE_35_SONNET_MODEL_NAME: ["claude3.5-sonnet", "c3.5", "c"],
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


def model_name_to_provider(model_name: str) -> Literal["anthropic", "openai"]:
    if model_name in ANTHROPIC_MODELS:
        return "anthropic"
    elif model_name in OPENAI_MODELS:
        return "openai"
    else:
        raise NotImplementedError(f"unrecognized {model_name}")
