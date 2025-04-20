#!/usr/bin/env bash
set -e

# optional
# python3 -m venv .venv
# source .venv/bin/activate

pip install -r requirements.txt

python -m src.train
