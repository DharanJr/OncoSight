"""
Local LLM backend for Module 5, using Ollama — free, runs entirely on your
GPU, no API key, no billing account, no internet required after setup.

One-time setup (outside Python):
    1. Download & install Ollama for Windows: https://ollama.com/download
    2. Pull the model once (downloads ~2GB, only needed the first time):
           ollama pull llama3.2:3b
    3. That's it — Ollama runs as a background service automatically after
       install, serving a local API at http://localhost:11434. Nothing else
       to start manually; if `ollama pull` worked, it's already running.

If Ollama isn't running or the model isn't pulled, functions here raise
OllamaUnavailableError, which generate_report.py catches to fall back to
the template-based generator instead of crashing.
"""

import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TIMEOUT_SECONDS


class OllamaUnavailableError(Exception):
    pass


def is_ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def call_ollama(prompt: str, max_tokens: int = 600) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()
    except requests.exceptions.ConnectionError:
        raise OllamaUnavailableError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama installed and running? See this file's docstring for setup."
        )
    except requests.exceptions.Timeout:
        raise OllamaUnavailableError(f"Ollama request timed out after {LLM_TIMEOUT_SECONDS}s.")
    except Exception as e:
        raise OllamaUnavailableError(f"Ollama call failed: {e}")