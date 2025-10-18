from typing import Literal


# OpenAI
GPT_41_MODEL_NAME = "gpt-4.1-2025-04-14"
O4_MINI_MODEL_NAME = "o4-mini-2025-04-16"
O3_MODEL_NAME = "o3-2025-04-16"
O3_DEEP_RESEARCH_MODEL_NAME = "o3-deep-research"
GPT_4_BASE = "gpt-4-base"
GPT_5_MODEL_NAME = "gpt-5-2025-08-07"

OPENAI_MODELS = [
    GPT_41_MODEL_NAME,
    O3_MODEL_NAME,
    O3_DEEP_RESEARCH_MODEL_NAME,
    O4_MINI_MODEL_NAME,
    GPT_4_BASE,
    GPT_5_MODEL_NAME,
]

# Anthropic models
CLAUDE_4_1_OPUS_MODEL_NAME = "claude-opus-4-1-20250805"
CLAUDE_4_5_SONNET_MODEL_NAME = "claude-sonnet-4-5-20250929"

ANTHROPIC_MODELS = [
    CLAUDE_4_1_OPUS_MODEL_NAME,
    CLAUDE_4_5_SONNET_MODEL_NAME,
]

# Google models

GEMINI_2_5_MODEL_NAME = "gemini-2.5-pro"

GOOGLE_MODELS = [GEMINI_2_5_MODEL_NAME]


GROK_4_MODEL_NAME = "grok-4-0709"
XAI_MODELS = [GROK_4_MODEL_NAME]

DEFAULT_MODEL_NAME = CLAUDE_4_1_OPUS_MODEL_NAME


def lacks_streaming_support(model_name: str) -> bool:
    assert not uses_legacy_completions(
        model_name
    ), f"Shouldn't be checking this for legacy model {model_name} -- shouldn't be using chat completions API"
    return False


def is_reasoning_model(model_name: str) -> bool:
    if (
        model_name == O4_MINI_MODEL_NAME
        or model_name == O3_MODEL_NAME
        or model_name == O3_DEEP_RESEARCH_MODEL_NAME
    ):
        return True
    else:
        return False


def supports_reasoning_effort(model_name: str) -> bool:
    """Returns True if model supports reasoning_effort parameter."""
    return model_name in [O4_MINI_MODEL_NAME, O3_MODEL_NAME, GPT_5_MODEL_NAME]


def supports_verbosity(model_name: str) -> bool:
    """Returns True if model supports verbosity parameter."""
    return model_name == GPT_5_MODEL_NAME


MODEL_NAME_TO_ABBREV = {
    GPT_41_MODEL_NAME: ["41", "4.1"],
    O4_MINI_MODEL_NAME: ["o4-mini", "o4m", "o4"],
    O3_MODEL_NAME: ["o3"],
    O3_DEEP_RESEARCH_MODEL_NAME: ["d", "o3d", "deep"],
    GPT_4_BASE: ["base"],
    GPT_5_MODEL_NAME: ["5", "gpt-5"],
    CLAUDE_4_1_OPUS_MODEL_NAME: ["c"],
    CLAUDE_4_5_SONNET_MODEL_NAME: ["cs"],
    GEMINI_2_5_MODEL_NAME: ["g"],
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
) -> Literal["anthropic", "openai", "openai_responses", "google", "xai"]:
    if model_name in ANTHROPIC_MODELS:
        return "anthropic"
    elif model_name == O3_DEEP_RESEARCH_MODEL_NAME:
        return "openai_responses"
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
