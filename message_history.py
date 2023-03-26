import pickle
from pathlib import Path
import colorama

# This provides functionality for saving and displaying message history.

LOG_FILENAME = "log.txt"
FILENAME_MESSAGE_HISTORY = "message_history.pkl"
PATHNAME_MESSAGE_HISTORY = "~/.gpt_cli/"

class History:

    def __init__(self, chat_name, system_prompt):
        self.chat_name = chat_name 
        self.message_history = [
            {"role": "system", "content": system_prompt}
        ]

    def get_message_history(self, for_openai = False):
        if for_openai:
            return [{"role" : line["role"], "content" : line["content"]} for line in self.message_history]
        else:
            return self.message_history
    
    def get_chat_name(self):
        return self.chat_name
    
    def append_user_message(self, user_prompt):
        self.message_history.append({"role": "user", "content": user_prompt})

    def append_response(self, response, model_name):
        self.message_history.append({"role": "assistant",
                                     "content": response,
                                     "model_name" : model_name})

    def display(self):
        pad_len = len("gpt-3.5-turbo:")
        for line in self.message_history:
            color = _get_line_color(line["role"])
            role_name = line["role"] if line["role"] != "assistant" else line["model_name"]
            role = (role_name+":").ljust(pad_len)
            content = line["content"]
            print(f"{color}{role} {content}")

def _get_line_color(role):
    if role == "user":
        color = colorama.Fore.BLUE 
    elif role == "assistant":
        color = colorama.Fore.BLACK 
    elif role == "system":
        color = colorama.Fore.RED 
    else:
        color = colorama.Fore.BLACK
    return color

def is_history_list(pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    path = Path(pathname).expanduser() / filename 
    return path.exists()

def init_history_list(pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    if not Path(pathname).expanduser().exists():
        Path(pathname).expanduser().mkdir()
    path = Path(pathname).expanduser() / filename 
    with open(path, "wb") as f:
        pickle.dump({"chat_names" : [], "history_list" : []}, f)

def get_chat_names(pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    path = Path(pathname).expanduser() / filename 
    with open(path, "rb") as f:
        d = pickle.load(f)
        return d["chat_names"]
    
def get_history_list(pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    path = Path(pathname).expanduser() / filename 
    with open(path, "rb") as f:
        d = pickle.load(f)
        return d["history_list"]
    
def can_append(history, pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    return history.get_chat_name() not in get_history_list(pathname, filename)

def append_history(history, pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    if can_append(history, pathname, filename):
        chat_names = get_chat_names(pathname, filename)
        history_list = get_history_list(pathname, filename)
        chat_names.append(history.get_chat_name())
        path = Path(pathname).expanduser() / filename 
        history_list.append(history)
        with open(path, "wb") as f:
            pickle.dump({
                "chat_names" : chat_names,
                "history_list" : history_list
            }, f)

def update_history(reply_index, user_prompt, response, model_name, pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    chat_names = get_chat_names(pathname, filename)
    history_list = get_history_list(pathname, filename)
    history_list[reply_index].append_user_message(user_prompt)
    history_list[reply_index].append_response(response, model_name)
    path = Path(pathname).expanduser() / filename 
    with open(path, "wb") as f:
        pickle.dump({
            "chat_names" : chat_names,
            "history_list" : history_list
        }, f)

def _display_history_line(chat_name, history):
    print(f"{chat_name[:20]}: {history.get_message_history()[1]['content'][:60]}...{history.get_message_history()[-1]['content'][-60:]}")

def display_history(index, pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    history_list = get_history_list(pathname, filename)
    history_list[index].display()

def display_all_history(pathname=PATHNAME_MESSAGE_HISTORY, filename=FILENAME_MESSAGE_HISTORY):
    chat_names = get_chat_names(pathname, filename)
    history_list = get_history_list(pathname, filename)
    for chat_name, history in zip(chat_names, history_list):
        _display_history_line(chat_name, history)

def write_to_log(content, pathname=PATHNAME_MESSAGE_HISTORY, filename=LOG_FILENAME):
    path = Path(pathname).expanduser() / filename
    with open(path, "a") as f:
        f.write(content)