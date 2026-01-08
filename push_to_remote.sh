#!/bin/bash

# Script to push cerebras_cli repository to a remote

if [ -z "$1" ]; then
    echo "Usage: $0 <remote_url>"
    echo "Example: $0 https://github.com/username/cerebras-cli.git"
    exit 1
fi

REMOTE_URL="$1"
REPO_DIR="/Users/shanks108/Development/cerebras_cli"

cd "$REPO_DIR" || exit 1

# Check if remote already exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "Remote 'origin' already exists. Updating..."
    git remote set-url origin "$REMOTE_URL"
else
    echo "Adding remote 'origin'..."
    git remote add origin "$REMOTE_URL"
fi

# Ensure we're on main branch
git branch -M main

# Push to remote
echo "Pushing to remote..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to $REMOTE_URL"
else
    echo "❌ Failed to push. Please check your remote URL and credentials."
    exit 1
fi
