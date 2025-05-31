# Simple command-line interface for LLMs

## Installation instructions 

1. Clone or download this repo

2. Install necessary dependencies:
```
pipenv install
```

3. Run `./install.sh`

4. Follow the instructions setting environment variables `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` and `XAI_API_KEY`

5. Run `ask "Hello world"`

## Basic usage

Display help:

`ask -h`

Make a query to Claude 4 Opus:

`ask "What is the fifth prime number?" -m c`

Display chat history: 

`ask -d`

Display history of the chat with ID 0:

`ask -d -c 0`

Reply to the last conversation: 

`ask "And the next one?" -r`

## Using file attachment feature

If you have additional context you would like to attach to your prompt (e.g. code or documentation), you can do that with these steps: 

**Step 1: write context to file**

`ask -w`

This will open a file `LLM_ATTACHED_CONTEXT.txt` in vim. Paste in the context you want, save and exit. 

**Step 2: query model**

`ask -f "How do I fix the following code?"`

When the `-f` flag is enabled, the text in `LLM_ATTACHED_CONTEXT.txt` will be appended to the end of your prompt. 

