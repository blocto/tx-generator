#!/bin/bash
# Check if .venv has been initiated
if [ ! -f .venv/bin/activate ]; then
    virtualenv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

pip install -r requirements.txt -q
poetry install

python case/downloader.py