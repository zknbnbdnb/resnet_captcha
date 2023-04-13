import os
from pathlib import Path

# This is Project Root
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Default save path for trained models
MODEL_PATH = ROOT_DIR / Path("model_save")
