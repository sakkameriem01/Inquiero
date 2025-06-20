import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
CHAT_HISTORY_DIR = ROOT_DIR / "chat_history"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

# PDF processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200 