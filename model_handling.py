GPT_4_MODEL_NAME = "gpt-4-0314"
GPT_3_MODEL_NAME = "gpt-3.5-turbo-0613"
GPT_4_32K_MODEL_NAME = "gpt-4-32k"
GPT_4_BASE_MODEL_NAME = "gpt-4-base"
DEFAULT_MODEL_NAME = GPT_4_MODEL_NAME


def uses_legacy_completions(model_name: str) -> bool:
    if model_name == GPT_4_BASE_MODEL_NAME:
        return True
    else:
        return False


def extract_model_name(model_short_str, default=DEFAULT_MODEL_NAME):
    """
    Extract the model name from a string input which was inputted via command line.
    Currently only works with gpt-3.5-turbo and gpt-4, because this is in chat mode.
    """
    if model_short_str == "3":
        model_name = GPT_3_MODEL_NAME
    elif model_short_str == "4":
        model_name = GPT_4_MODEL_NAME
    elif model_short_str == "32k":
        model_name = GPT_4_32K_MODEL_NAME
    elif model_short_str == "base":
        model_name = GPT_4_BASE_MODEL_NAME
    else:
        model_name = default
    return model_name
