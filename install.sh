#!/bin/bash

# Make directory for storing log files and message history
if [ ! -d ~/.gpt_cli ]; then
    echo "Creating ~/.gpt_cli directory for storing log files and message history"
    mkdir ~/.gpt_cli
fi

# Make ~/bin if doesn't exist
if [ ! -d ~/bin ]; then
    echo "Creating ~/bin directory for storing scripts"
    mkdir ~/bin
fi

# Create ask symlink in ~/bin to gpt_cli.py
if [ ! -f ~/bin/ask ]; then
    echo "Creating symlink ~/bin/ask to gpt_cli.py"
    ln -s $(pwd)/gpt_cli.py ~/bin/ask
fi

# Add ~/bin to $PATH environment variable if not already there
if [[ ! "$PATH" =~ "$HOME/bin" ]]; then
    echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc
fi

echo "Successfully installed.

Before using, you still need to add these lines to ~/.bashrc:
export OPENAI_API_KEY     =<your API key>
export ANTHROPIC_API_KEY  =<your API key>
export GOOGLE_API_KEY     =<your API key>
export XAI_API_KEY        =<your API key>
Find this information at https://platform.openai.com/account/api-keys and https://platform.openai.com/account/org-settings and https://console.anthropic.com/settings/keys and https://aistudio.google.com/app/apikey

Then run 
source ~/.bashrc"
