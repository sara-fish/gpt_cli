import pickle
from pathlib import Path
import colorama
from shutil import get_terminal_size
from model_handling import uses_legacy_completions, GPT_4_BASE_MODEL_NAME

# This provides functionality for saving and displaying message history.

LOG_FILENAME = "log.txt"
FILENAME_MESSAGE_HISTORY = "message_history.pkl"
PATHNAME_MESSAGE_HISTORY = "~/.gpt_cli/"

USER_COLOR = colorama.Fore.BLUE
SYSTEM_COLOR = colorama.Fore.RED
ASSISTANT_COLOR = colorama.Style.RESET_ALL
DEFAULT_COLOR = colorama.Style.RESET_ALL

colorama.init(autoreset=True)


class History:
    def __init__(self, chat_name, system_prompt, legacy):
        self.chat_name = chat_name
        self.legacy = legacy
        if self.legacy:
            self.message_history = []
        else:
            self.message_history = [{"role": "system", "content": system_prompt}]

    def is_legacy(self):
        try:
            return self.legacy
        except AttributeError:
            # Older part of message history, prior to introduction of this feature.
            self.legacy = False
            return self.legacy

    def get_message_history(self, for_openai=False, legacy=False):
        if for_openai:
            if not legacy:
                return [
                    {"role": line["role"], "content": line["content"]}
                    for line in self.message_history
                ]
            else:
                # Must convert to string
                prompt = ""
                for line in self.message_history:
                    prompt += line["content"]
                return prompt
        else:
            return self.message_history

    def get_chat_name(self):
        return self.chat_name

    def append_user_message(self, user_prompt):
        self.message_history.append({"role": "user", "content": user_prompt})

    def append_response(self, response, model_name):
        self.message_history.append(
            {"role": "assistant", "content": response, "model_name": model_name}
        )

    def _compute_pad_len(self):
        models_in_conversation = {"system"}
        for line in self.message_history:
            if line["role"] == "assistant":
                models_in_conversation.add(line["model_name"])
        return max(map(len, models_in_conversation)) + 1

    def display(self):
        if not self.is_legacy():
            pad_len = self._compute_pad_len()
            for line in self.message_history:
                color = _get_line_color(line["role"])
                role_name = (
                    line["role"] if line["role"] != "assistant" else line["model_name"]
                )
                role = (role_name + ":").ljust(pad_len)
                content = line["content"]
                print(f"{color}{role} {content}")
        else:
            for line in self.message_history:
                color = _get_line_color(line["role"])
                content = line["content"]
                print(f"{color}{content}", end="")
            print()


def _get_line_color(role):
    if role == "user":
        color = USER_COLOR
    elif role == "assistant":
        color = ASSISTANT_COLOR
    elif role == "system":
        color = SYSTEM_COLOR
    else:
        color = DEFAULT_COLOR
    return color


def is_history_list(
    pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY
):
    path = Path(pathname).expanduser() / filename
    return path.exists()


def init_history_list(
    pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY
):
    if not Path(pathname).expanduser().exists():
        Path(pathname).expanduser().mkdir()
    path = Path(pathname).expanduser() / filename
    with open(path, "wb") as f:
        pickle.dump({"chat_names": [], "history_list": []}, f)


def get_chat_names(
    pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY
):
    path = Path(pathname).expanduser() / filename
    with open(path, "rb") as f:
        d = pickle.load(f)
        return d["chat_names"]


def get_history_list(
    pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY
):
    path = Path(pathname).expanduser() / filename
    with open(path, "rb") as f:
        d = pickle.load(f)
        return d["history_list"]


def can_append(
    history, pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY
):
    return history.get_chat_name() not in get_history_list(pathname, filename)


def append_history(
    history, pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY
):
    if can_append(history, pathname, filename):
        chat_names = get_chat_names(pathname, filename)
        history_list = get_history_list(pathname, filename)
        chat_names.append(history.get_chat_name())
        path = Path(pathname).expanduser() / filename
        history_list.append(history)
        with open(path, "wb") as f:
            pickle.dump({"chat_names": chat_names, "history_list": history_list}, f)


def update_history(
    reply_index,
    user_prompt,
    response,
    model_name,
    pathname=PATHNAME_MESSAGE_HISTORY,
    filename=FILENAME_MESSAGE_HISTORY,
):
    chat_names = get_chat_names(pathname, filename)
    history_list = get_history_list(pathname, filename)
    history_list[reply_index].append_user_message(user_prompt)
    history_list[reply_index].append_response(response, model_name)
    path = Path(pathname).expanduser() / filename
    with open(path, "wb") as f:
        pickle.dump({"chat_names": chat_names, "history_list": history_list}, f)


def _display_history_line(chat_name, history):
    line_length = get_terminal_size().columns
    start_length = line_length // 2 - 2 - len(chat_name)
    end_length = line_length // 2 - 2 - len(chat_name)
    start_idx = 0 if history.is_legacy() else 1
    start_message = f"{USER_COLOR}{history.get_message_history()[start_idx]['content'][:start_length]}".replace(
        "\n", ""
    ).replace(
        "\r", ""
    )
    end_message = f"{ASSISTANT_COLOR}{history.get_message_history()[-1]['content'][-end_length:]}".replace(
        "\n", ""
    ).replace(
        "\r", ""
    )
    print(f"{chat_name}: {start_message}...{end_message}")


def display_history(
    index, pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY
):
    history_list = get_history_list(pathname, filename)
    history_list[index].display()


def display_all_history(
    pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY
):
    chat_names = get_chat_names(pathname, filename)
    history_list = get_history_list(pathname, filename)
    for chat_name, history in zip(chat_names, history_list):
        _display_history_line(chat_name, history)


def write_to_log(content, pathname=PATHNAME_MESSAGE_HISTORY, filename=LOG_FILENAME):
    path = Path(pathname).expanduser() / filename
    with open(path, "a") as f:
        f.write(content)
