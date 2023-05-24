# Simple command-line interface for OpenAI's GPT-3.5/4

## Installation instructions 

1. Clone or download this repo

2. `pip install openai` and `pip install colorama`

3. Run `./install.sh`

4. Follow the instructions about OPENAI_API_KEY 

5. Run `gpt "Hello world"`

## How to use 

Display help:

`gpt -h`

Make a query to GPT-3.5: 

`gpt "What is the fifth prime number?" -m 3`

Make a query to GPT-4:

`gpt "What is the fifth prime number?" -m 4`

Display chat history: 

`gpt -d`

Display history of the chat with ID 0:

`gpt -d -c 0`

Reply to the last conversation: 

`gpt "And the next one?" -r`

### How to use file attachment feature 

If you have additional context you would like to attach to your prompt (e.g. code or documentation), you can do that with these steps: 

**Step 1: write context to file**

`gpt -w`

This will open a file `GPT_ATTACHED_CONTEXT.txt` in vim. Paste in the context you want, save and exit. 

**Step 2: query model**

`gpt -f "How do I fix the following code?"`

When the `-f` flag is enabled, the text in `GPT_ATTACHED_CONTEXT.txt` will be appended to the end of your prompt. 
