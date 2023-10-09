# Simple command-line interface for OpenAI's GPT-3.5/4

## Installation instructions 

1. Clone or download this repo

2. `pip install openai` and `pip install colorama`

3. Run `./install.sh`

4. Follow the instructions about `OPENAI_API_KEY`

5. Run `gpt "Hello world"`

## Basic usage

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

## Using file attachment feature

If you have additional context you would like to attach to your prompt (e.g. code or documentation), you can do that with these steps: 

**Step 1: write context to file**

`gpt -w`

This will open a file `GPT_ATTACHED_CONTEXT.txt` in vim. Paste in the context you want, save and exit. 

**Step 2: query model**

`gpt -f "How do I fix the following code?"`

When the `-f` flag is enabled, the text in `GPT_ATTACHED_CONTEXT.txt` will be appended to the end of your prompt. 


## Using different OpenAI organization ids for different models 

By default, all queries will use the same organization id, `OPENAI_ORGANIZATION`. To override this behavior for specific models, you can add to your `.bashrc` any subset of the following lines.

```
export OPENAI_ORGANIZATION_3=<org id for gpt-3.5-turbo> # this org id will be used for all gpt-3.5-turbo calls
export OPENAI_ORGANIZATION_4=<org id for gpt-4>         # this org id will be used for all gpt-4 calls 
export OPENAI_ORGANIZATION_32=<org id for gpt-4-32k>    # this org id will be used for all gpt-4-32k calls
export OPENAI_ORGANIZATION_base=<org id for gpt-4-base> # this org id will be used for all gpt-4-base calls 
```