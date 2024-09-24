#!/bin/bash

# Check if 'tree' is installed, if not, fall back to 'find' command
if command -v tree &> /dev/null; then
    echo "Using 'tree' command to display the file structure in the current directory..."
    tree -C -I ".*"
else
    echo "'tree' command is not installed. Using 'find' instead..."
    # Using 'find' to display file structure in the current directory without hidden files
    find . -path './.*' -prune -o -print | sed -e 's;[^/]*/;|___;g;s;___|; |;g'
fi

