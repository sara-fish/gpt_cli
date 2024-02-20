#!/bin/bash

# Make directory for storing log files and message history
if [ ! -d ~/.gpt_cli ]; then
    mkdir ~/.gpt_cli
fi

# Make ~/bin if doesn't exist
if [ ! -d ~/bin ]; then
    mkdir ~/bin
fi

# Create ask symlink in ~/bin to gpt_cli.py
if [ ! -f ~/bin/ask ]; then
    ln -s $(pwd)/gpt_cli.py ~/bin/ask
fi

# Add ~/bin to $PATH environment variable if not already there
if [[ ! "$PATH" =~ "$HOME/bin" ]]; then
    echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc
fi

echo "Successfully installed.

Before using, you still need to add these lines to ~/.bashrc:
export OPENAI_API_KEY     =<your API key>
export OPENAI_ORGANIZATION=<your organization ID>
Find this information at https://platform.openai.com/account/api-keys and https://platform.openai.com/account/org-settings

Then run 
source ~/.bashrc"
