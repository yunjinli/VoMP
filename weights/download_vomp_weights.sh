#!/bin/bash

set -e

REPO_URL="https://huggingface.co/nvidia/PhysicalAI-Simulation-VoMP-Model"
TARGET_DIR="weights/vomp"

if ! command -v git-lfs &> /dev/null; then
    echo "ERROR: git-lfs is not installed."
    exit 1
fi

echo "Cloning repository from $REPO_URL to $TARGET_DIR..."
git clone $REPO_URL $TARGET_DIR

mv $TARGET_DIR/* weights/
echo "Cleaning up..."
rm -rf $TARGET_DIR