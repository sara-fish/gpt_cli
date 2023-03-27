# gpt_cli

## Installation instructions 

1. Clone or download this repo

2. `pip3 install openai` and `pip3 install colorama`

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