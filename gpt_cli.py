#!/bin/python3

import os
import openai
import argparse
from datetime import datetime
import subprocess

import message_history

DEFAULT_MODEL_NAME = "gpt-3.5-turbo"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate AI assistant who provides BRIEF excellent responses to queries (NEVER say fluff like 'As an AI')."

DEFAULT_FILENAME = "GPT_ATTACHED_CONTEXT.txt"


def extract_model_name(input, default=DEFAULT_MODEL_NAME):
    """
    Extract the model name from a string input which was inputted via command line.
    Currently only works with gpt-3.5-turbo and gpt-4, because this is in chat mode.
    """
    if "3" in input:
        model_name = "gpt-3.5-turbo"
    elif "4" in input:
        model_name = "gpt-4"
    else:
        model_name = default
    return model_name


if __name__ == "__main__":
    # Parse command line input

    # Set up parser
    parser = argparse.ArgumentParser(
        description="Simple command line interface for OpenAI API."
    )
    parser.add_argument(
        "prompt", nargs="?", type=str, help='Your prompt (e.g. "Hello World").'
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display entire usage history or (if conversation ID given) a specific conversation's history",
    )
    parser.add_argument(
        "-r",
        "--reply",
        action="store_true",
        help="Reply to previous conversation (default is most recent, or select with conversation ID)",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs=1,
        default=DEFAULT_MODEL_NAME,
        type=str,
        help="Model to use ('3' for gpt-3.5-turbo, '4' for gpt-4)",
    )
    parser.add_argument(
        "-c",
        "--conversation_id",
        type=int,
        help="Specify which conversation to reply to or display",
    )
    parser.add_argument(
        "-s",
        "--system",
        nargs="?",
        default=DEFAULT_SYSTEM_PROMPT,
        type=str,
        help="System prompt to use",
    )
    parser.add_argument(
        "-f",
        "--fileread",
        action="store_true",
        help=f"If selected, appends to prompt content from {DEFAULT_FILENAME}",
    )
    parser.add_argument(
        "-w",
        "--filewrite",
        action="store_true",
        help=f"If selected, opens {DEFAULT_FILENAME} in vim to let you paste content in.",
    )

    # Parse and extract args
    args = parser.parse_args()

    user_prompt = args.prompt
    reply_mode = args.reply
    display_mode = args.display
    model_name = extract_model_name(args.model)
    conv_id = args.conversation_id
    system_prompt = args.system
    fileread = args.fileread
    filewrite = args.filewrite

    # First handle display mode
    if display_mode:
        if conv_id != None:
            message_history.display_history(conv_id)
        else:
            message_history.display_all_history()
        exit(0)

    # Next handle write mode
    if filewrite:
        subprocess.run(["vim", DEFAULT_FILENAME])
        exit(0)

    # Otherwise enter conversation mode

    if user_prompt == None:
        parser.print_help()
        exit(1)

    # If history list does not exist, create it
    # (i.e. there is no message_history.pkl saved where you expect)
    if not message_history.is_history_list():
        message_history.init_history_list()

    chat_names = message_history.get_chat_names()
    history_list = message_history.get_history_list()

    # Get current chat name and history
    if reply_mode:
        # If conv_id specified, reply to that, otherwise reply to most recent conversation
        if conv_id != None:
            reply_index = conv_id
        else:
            reply_index = -1
        try:
            current_history = history_list[reply_index]
            current_chat_name = current_history.get_chat_name()
        except IndexError:
            print("Can't reply to empty history.")
    else:
        current_chat_name = str(len(chat_names))
        current_history = message_history.History(current_chat_name, system_prompt)

    if fileread:
        try:
            with open(DEFAULT_FILENAME, "r") as f:
                fileread_content = f.read()
        except FileNotFoundError:
            print(
                f"Error: {DEFAULT_FILENAME} not found. First run `gpt -w` to write to this file."
            )
            exit(1)
        full_prompt = user_prompt + " Attached context: " + fileread_content
    else:
        full_prompt = user_prompt

    current_history.append_user_message(full_prompt)

    # Talk to model

    openai.organization = os.getenv("OPENAI_ORGANIZATION")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    completion = openai.ChatCompletion.create(
        model=model_name, messages=current_history.get_message_history(for_openai=True)
    )

    response = completion.choices[0].message.content

    # Print to terminal
    print(response)

    # Log to history
    if reply_mode:
        message_history.update_history(reply_index, user_prompt, response, model_name)
    else:
        # Add GPT's response to current_history object
        current_history.append_response(response, model_name)
        # Save current_history object to message history
        message_history.append_history(current_history)

    # Log all raw data

    message_history.write_to_log(
        f"time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
    )
    message_history.write_to_log(f"prompt: {user_prompt}\n")
    message_history.write_to_log(str(completion) + "\n\n")
