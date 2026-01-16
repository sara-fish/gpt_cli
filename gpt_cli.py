#!/usr/bin/env python3

import os
from openai import OpenAI
import anthropic
from google import genai
import argparse
from datetime import datetime
import subprocess
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

import message_history
from model_handling import (
    MODEL_NAME_TO_ABBREV,
    extract_model_name,
    lacks_streaming_support,
    MODEL_NAME_TO_ABBREV_LEGEND,
    model_name_to_provider,
    DEFAULT_MODEL_NAME,
    uses_legacy_completions,
    supports_reasoning_effort,
    supports_verbosity,
)

CLAUDE_PROMPT_EXCERPT = """
You should give concise responses to very simple questions, but provide thorough responses to complex and open-ended questions. You should discuss virtually any topic factually and objectively. You should explain difficult concepts or ideas clearly. You should also illustrate your explanations with examples, thought experiments, or metaphors. If the user corrects you or tells you that you've made a mistake, you should first think through the issue carefully before acknowledging the user, since users sometimes make errors themselves. You should never start your response by saying a question or idea or observation was good, great, fascinating, profound, excellent, or any other positive adjective. You should skip the flattery and respond directly. You should critically evaluate any theories, claims, and ideas presented to you rather than automatically agreeing or praising them. When presented with dubious, incorrect, ambiguous, or unverifiable theories, claims, or ideas, you should respectfully point out flaws, factual errors, lack of evidence, or lack of clarity rather than validating them. You should prioritize truthfulness and accuracy over agreeability, and should not tell people that incorrect theories are true just to be polite. When engaging with metaphorical, allegorical, or symbolic interpretations (such as those found in continental philosophy, religious texts, literature, or psychoanalytic theory), you should acknowledge their non-literal nature while still being able to discuss them critically.
""".strip()

USER_INFO = """
- You can assume the user has PhD-level knowledge in mathematics, computer science, and economics.
- When the user asks you about math, give intuition and then be rigorous (using formulas/equations when needed).
- When the user asks you to write code, write the code in one big block. Just write the code and nothing else -- no explanation needed (unless requested otherwise).
- When the user asks for writing advice, give multiple options, and use academic language (unless requested otherwise). Unless specified otherwise you can assume the reader is asking for help with writing an academic paper for a CS conference or econ journal.
""".strip()

DEFAULT_SYSTEM_PROMPT = CLAUDE_PROMPT_EXCERPT + USER_INFO

DEFAULT_FILENAME = "LLM_ATTACHED_CONTEXT.txt"


if __name__ == "__main__":
    # Parse command line input

    # Set up parser
    parser = argparse.ArgumentParser()
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
        "-p",
        "--private",
        action="store_true",
        help="Use private mode (no logging to history).",
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
        nargs="?",
        default=DEFAULT_MODEL_NAME,
        type=str,
        help=f"Model to use: {MODEL_NAME_TO_ABBREV_LEGEND}",
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
        help=f'System prompt to use. Default: "{DEFAULT_SYSTEM_PROMPT}"',
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

    parser.add_argument(
        "-t", "--temperature", help="Set the temperature for the query.", type=float
    )

    parser.add_argument(
        "-re",
        "--reasoning",
        nargs="?",
        default="high",
        type=str,
        choices=["minimal", "low", "medium", "high"],
        help='Reasoning effort level: "minimal", "low", "medium", "high" (default: high)',
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        nargs="?",
        default="low",
        type=str,
        choices=["low", "medium", "high"],
        help='Verbosity level: "low", "medium", "high" (default: low)',
    )

    parser.add_argument(
        "--disable-markdown",
        action="store_true",
        help="Disable markdown rendering and print raw output",
    )

    # Parse and extract args
    args = parser.parse_args()

    user_prompt = args.prompt
    reply_mode = args.reply
    display_mode = args.display
    short_model_name = args.model
    model_name = extract_model_name(short_model_name)
    conv_id = args.conversation_id
    system_prompt = args.system
    fileread = args.fileread
    filewrite = args.filewrite
    temperature = args.temperature
    reasoning_effort = args.reasoning
    verbosity = args.verbosity
    disable_markdown = args.disable_markdown

    # First handle display mode
    if display_mode:
        if conv_id is not None:
            message_history.display_history(conv_id)
        else:
            message_history.display_all_history()
        exit(0)

    # Next handle write mode
    if filewrite:
        subprocess.run(["vim", DEFAULT_FILENAME])
        exit(0)

    # Otherwise enter conversation mode

    if user_prompt is None:
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
        if conv_id is not None:
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
        current_history = message_history.History(
            current_chat_name,
            system_prompt,
            legacy=uses_legacy_completions(model_name),
        )

    if fileread:
        try:
            with open(DEFAULT_FILENAME, "r") as f:
                fileread_content = f.read()
        except FileNotFoundError:
            print(
                f"Error: {DEFAULT_FILENAME} not found. First run `gpt -w` to write to this file."
            )
            exit(1)
        full_prompt = user_prompt + fileread_content
    else:
        full_prompt = user_prompt

    current_history.append_user_message(full_prompt)

    # Build optional arguments for API calls
    optional_args = {}
    if temperature is not None:
        optional_args["temperature"] = temperature

    # Add reasoning_effort for supported models (o3, o4-mini, GPT-5)
    if supports_reasoning_effort(model_name) and reasoning_effort is not None:
        optional_args["reasoning_effort"] = reasoning_effort

    # Add verbosity for supported models (GPT-5)
    if supports_verbosity(model_name) and verbosity is not None:
        optional_args["verbosity"] = verbosity

    # Talk to model

    provider = model_name_to_provider(model_name)

    console = Console()  # for printing markdown

    if provider == "anthropic":
        try:
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY_CLI"))

            system_prompt, messages = current_history.get_message_history(
                platform="anthropic"
            )

            if system_prompt is None:
                messages_dict = {"messages": messages}
            else:
                messages_dict = {"system": system_prompt, "messages": messages}

            response = ""
            buffer = ""
            usage = None

            with client.messages.stream(
                model=model_name, max_tokens=4000, **messages_dict, **optional_args
            ) as stream:
                if disable_markdown:
                    for text in stream.text_stream:
                        response += text
                        print(text, end="", flush=True)
                else:
                    with Live(console=console, refresh_per_second=10) as live:
                        for text in stream.text_stream:
                            response += text
                            live.update(Markdown(response))

                # Get usage info
                final_message = stream.get_final_message()
                usage = final_message.usage

        except KeyboardInterrupt:
            print("<KeyboardInterrupt>", flush=True)
        else:
            print()
            # Print token usage for Anthropic
            if usage is not None:
                print(
                    f"Input tokens: {usage.input_tokens} Output tokens: {usage.output_tokens}"
                )

    elif provider == "google":
        try:
            system_prompt, messages = current_history.get_message_history(
                platform="google"
            )

            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_CLI"))

            completion = client.models.generate_content_stream(
                model=model_name,
                contents=messages,
            )

            response = ""
            usage_metadata = None

            if disable_markdown:
                for chunk in completion:
                    if chunk.text:
                        response += chunk.text
                        print(chunk.text, end="", flush=True)
                    if hasattr(chunk, "usage_metadata"):
                        usage_metadata = chunk.usage_metadata
            else:
                with Live(console=console, refresh_per_second=10) as live:
                    for chunk in completion:
                        if chunk.text:
                            response += chunk.text
                        live.update(Markdown(response))
                        if hasattr(chunk, "usage_metadata"):
                            usage_metadata = chunk.usage_metadata

        except KeyboardInterrupt:
            print("<KeyboardInterrupt>", flush=True)
        else:
            print()
            # Print token usage for Google
            if usage_metadata is not None:
                print(
                    f"Input tokens: {usage_metadata.prompt_token_count} Output tokens: {usage_metadata.candidates_token_count}"
                )

    elif provider == "openai_responses":
        # Handle o3-deep-research using Responses API
        api_key = os.getenv("OPENAI_API_KEY_CLI")

        # Check for model-specific API key
        model_abbrevs = MODEL_NAME_TO_ABBREV.get(model_name, [])
        for model_abbrev in model_abbrevs:
            env_var = f"OPENAI_API_KEY_{model_abbrev}"
            if os.getenv(env_var):
                api_key = os.getenv(env_var)
                break

        client = OpenAI(api_key=api_key)

        try:
            messages = current_history.get_message_history(platform="openai")

            # Convert messages to the input format for responses API
            # The responses API expects an input parameter with conversation
            input_messages = []
            for msg in messages:
                # For text-only messages, use the standard format
                input_messages.append({"role": msg["role"], "content": msg["content"]})

            # Create request with optional temperature
            request_params = {
                "model": model_name,
                "input": input_messages,
                # o3-deep-research requires at least one tool
                "tools": [{"type": "web_search_preview"}],
                "parallel_tool_calls": True,
            }
            if temperature is not None:
                request_params["temperature"] = temperature
            # o3-deep-research only supports 'medium' reasoning effort, not 'high'
            # Don't pass reasoning effort parameter - let it use the default
            # Note: verbosity is not supported by o3-deep-research in Responses API

            # Use responses.create API with streaming
            request_params["stream"] = True
            stream = client.responses.create(**request_params)

            response = ""
            usage_obj = None

            for event in stream:
                try:
                    # Handle different event types
                    if hasattr(event, "type"):
                        if event.type == "response.output_text.delta":
                            if hasattr(event, "delta") and event.delta:
                                response += event.delta
                        elif event.type == "response.output.delta":
                            if hasattr(event, "delta") and event.delta:
                                response += str(event.delta)
                        elif event.type == "response.completed":
                            if hasattr(event, "response"):
                                # Extract usage from completed event
                                if hasattr(event.response, "usage"):
                                    usage_obj = event.response.usage

                    # Fallback: try to extract text from event
                    if hasattr(event, "text") and event.text:
                        response += event.text
                    elif hasattr(event, "output_text") and event.output_text:
                        response += event.output_text
                except Exception as e:
                    print(f"\nDEBUG - Event error: {e}")
                    print(f"DEBUG - Event object: {event}")
                    raise

            # Display response
            if response:
                console.print(Markdown(response))
            else:
                print("No response received")

            completion = usage_obj  # For usage reporting below

        except KeyboardInterrupt:
            print("<KeyboardInterrupt>", flush=True)
        else:
            print()
            # Print token usage for Responses API
            if hasattr(completion, "usage"):
                usage = completion.usage
                tokens_str = f"Input tokens: {usage.input_tokens} Output tokens: {usage.output_tokens}"
                if hasattr(usage, "reasoning_tokens") and usage.reasoning_tokens:
                    tokens_str += f" Reasoning tokens: {usage.reasoning_tokens}"
                print(tokens_str)

    elif provider == "openai" or provider == "xai":
        base_url = "https://api.x.ai/v1" if provider == "xai" else None
        if provider == "xai":
            api_key = os.getenv("XAI_API_KEY_CLI")
        else:
            # first detect if there's a model-specific API key
            model_abbrevs = MODEL_NAME_TO_ABBREV.get(model_name, [])
            for model_abbrev in model_abbrevs:
                env_var = f"OPENAI_API_KEY_{model_abbrev}"
                if os.getenv(env_var):
                    api_key = os.getenv(env_var)
                    break
            else:
                api_key = os.getenv("OPENAI_API_KEY_CLI")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        usage = None
        try:
            if uses_legacy_completions(model_name):
                prompt = current_history.get_message_history(platform="legacy")

                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    stream=True,
                    max_tokens=4000,
                    **optional_args,
                )  # type: ignore

                response = ""
                for chunk in completion:
                    chunk_message_str = chunk.choices[0].text
                    response += chunk_message_str
                    print(chunk_message_str, end="", flush=True)

            elif lacks_streaming_support(model_name):
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=current_history.get_message_history(platform="openai"),
                    **optional_args,
                )  # type: ignore

                response = completion.choices[0].message.content
                print(response)
                usage = completion.usage

            else:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=current_history.get_message_history(platform="openai"),
                    stream=True,
                    **optional_args,
                    stream_options={"include_usage": True},
                )  # type: ignore

                response = ""
                if disable_markdown:
                    for chunk in completion:
                        try:
                            if chunk.usage:
                                usage = chunk.usage
                            if len(chunk.choices) > 0:
                                chunk_message = chunk.choices[0].delta
                                chunk_message_str = (
                                    chunk_message.content
                                    if chunk_message.content is not None
                                    else ""
                                )
                                response += chunk_message_str
                                print(chunk_message_str, end="", flush=True)
                        except Exception as e:
                            print(f"\nDEBUG - Chunk error: {e}")
                            print(f"DEBUG - Chunk object: {chunk}")
                            raise
                else:
                    with Live(console=console, refresh_per_second=10) as live:
                        for chunk in completion:
                            try:
                                if chunk.usage:
                                    usage = chunk.usage
                                if len(chunk.choices) > 0:
                                    chunk_message = chunk.choices[0].delta
                                    chunk_message_str = (
                                        chunk_message.content
                                        if chunk_message.content is not None
                                        else ""
                                    )
                                    response += chunk_message_str
                                    live.update(Markdown(response))
                            except Exception as e:
                                print(f"\nDEBUG - Chunk error: {e}")
                                print(f"DEBUG - Chunk object: {chunk}")
                                raise

        except KeyboardInterrupt:
            print("<KeyboardInterrupt>", flush=True)
        else:
            print()
            # Print token usage for OpenAI/XAI
            if usage is not None:
                tokens_str = f"Input tokens: {usage.prompt_tokens} Output tokens: {usage.completion_tokens}"
                if (
                    hasattr(usage, "completion_tokens_details")
                    and usage.completion_tokens_details
                ):
                    details = usage.completion_tokens_details
                    if (
                        hasattr(details, "reasoning_tokens")
                        and details.reasoning_tokens
                    ):
                        tokens_str += f" Reasoning tokens: {details.reasoning_tokens}"
                print(tokens_str)

    # Log to history
    if args.private:
        pass
    else:
        if reply_mode:
            message_history.update_history(
                reply_index, user_prompt, response, model_name
            )
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
        message_history.write_to_log(f"model: {model_name}\n")
        message_history.write_to_log(f"response: {response}\n\n")
        if provider == "google":
            message_history.write_to_log(str(completion) + "\n\n")
