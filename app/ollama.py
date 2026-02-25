"""
BenchDrift App — LLM backend helpers.

Thin wrappers over model_client.py clients. No duplicate HTTP logic.
All actual HTTP calls go through the pipeline's client classes:
  - OllamaClient (native /api/chat + OpenAI-compat /v1/chat/completions)
  - GroqClient (OpenAI-compat /chat/completions)

This module provides:
  - call_llm(): single-call dispatcher used by app/runner.py
  - get_available_models(): model listing for UI dropdowns
  - Response cleaning functions (app-specific, not client logic)
"""

import os
import re
from typing import List

from benchdrift.pipeline.comprehensive_variation_engine_v2 import clean_model_response
from benchdrift.models.model_client import (
    GroqClient,
    ModelClientFactory,
    OllamaClient,
    OLLAMA_BASE_URL,
)

# Re-export for backward compatibility
OLLAMA_BASE_URL = OLLAMA_BASE_URL

# ---------------------------------------------------------------------------
# Client cache — one instance per (backend, model, base_url) tuple
# ---------------------------------------------------------------------------
_client_cache = {}


def _get_client(backend: str, model: str, base_url: str = "", timeout: int = 0):
    """Get or create a cached client instance."""
    cache_key = (backend, model, base_url)
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    if backend == "groq":
        client = GroqClient(model_name=model, max_workers=1, max_new_tokens=1024)
    else:
        client = OllamaClient(model_name=model, max_workers=1, max_new_tokens=1024)
        if base_url:
            client.base_url = base_url

    _client_cache[cache_key] = client
    return client


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------
BACKENDS = ["ollama", "groq"]


# Ollama cloud models — run on Ollama's cloud infra, no local GPU needed
_CLOUD_OLLAMA_MODELS = [
    "qwen3-coder-next:cloud", "qwen3.5:cloud", "qwen3-next:cloud",
    "deepseek-v3.2:cloud", "kimi-k2.5:cloud", "kimi-k2-thinking:cloud",
    "glm-5:cloud", "glm-4.7:cloud", "glm-4.6:cloud",
    "minimax-m2.5:cloud", "minimax-m2.1:cloud", "minimax-m2:cloud",
    "gemini-3-flash-preview:cloud",
    "devstral-2:cloud", "devstral-small-2:cloud",
    "cogito-2.1:cloud", "nemotron-3-nano:cloud", "rnj-1:cloud",
]


def get_available_models(backend: str = "ollama", base_url: str = "") -> List[str]:
    """Get models for a single backend (bare names, no prefix).

    For Ollama: returns locally pulled models + cloud models.
    No 'available to pull' noise — only what's on disk or in the cloud.
    """
    if backend == "groq":
        models = GroqClient.get_available_models()
        from benchdrift.models.model_client import GROQ_API_KEY
        return models if (models and GROQ_API_KEY) else []

    # default: ollama
    local_models = []
    try:
        import requests
        url = base_url or OLLAMA_BASE_URL
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        local_models = sorted(m["name"] for m in resp.json().get("models", []))
    except Exception:
        pass

    result = local_models
    if _CLOUD_OLLAMA_MODELS:
        result += _CLOUD_OLLAMA_MODELS
    return result


def get_merged_models(backends: List[str], base_url: str = "") -> List[str]:
    """Get models from all selected backends, prefixed as 'client/model'.

    Returns a flat list like:
        ["ollama/qwen3:8b", "ollama/mistral:7b", "groq/llama-3.3-70b", ...]
    """
    merged = []
    for backend in backends:
        models = get_available_models(backend, base_url)
        for m in models:
            merged.append(f"{backend}/{m}")
    return merged if merged else ["(no models — check backends)"]


def parse_model_selection(selection: str) -> tuple:
    """Parse 'client/model' dropdown value into (backend, model_name).

    Returns:
        ("ollama", "qwen3:8b") for "ollama/qwen3:8b"
        ("ollama", selection) as fallback for bare names
    """
    if not selection:
        return "ollama", ""
    for b in BACKENDS:
        prefix = f"{b}/"
        if selection.startswith(prefix):
            return b, selection[len(prefix):]
    # Fallback: bare model name → assume ollama
    return "ollama", selection


def call_llm(model: str, system_prompt: str, user_prompt: str,
             backend: str = "ollama", max_tokens: int = 1024,
             temperature: float = 0.5, think: bool = False,
             base_url: str = "", timeout: int = 0,
             return_reasoning: bool = False):
    """Unified single LLM call — routes to the right backend client.

    This is the ONLY function app/runner.py and app/hf_loader.py should call.
    All actual HTTP logic lives in model_client.py client classes.

    Args:
        return_reasoning: If True and think=True, returns dict {"content": str, "thinking": str}
                          instead of just the content string.
    Returns:
        str normally, or dict {"content": str, "thinking": str} if return_reasoning=True
    """
    client = _get_client(backend, model, base_url, timeout)

    if backend == "groq":
        result = client.call_groq_llm(
            system_prompt, user_prompt,
            max_new_tokens=max_tokens, temperature=temperature,
            timeout=timeout or 60,
        )
        if return_reasoning:
            return {"content": result, "thinking": ""}
        return result

    # Ollama — use native /api/chat for think support
    return client.call_native(
        system_prompt, user_prompt,
        max_new_tokens=max_tokens, temperature=temperature,
        think=think, timeout=timeout or 120,
        return_reasoning=return_reasoning,
    )


# ---------------------------------------------------------------------------
# Response cleaning (app-specific, not client logic)
# ---------------------------------------------------------------------------
def clean_think_tags(text: str) -> str:
    """Strip <think> tags from solver responses."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def clean_ollama_response(raw: str) -> str:
    """Full response cleaning: strip <think>, extract <question>, strip preamble."""
    if not raw:
        return ""
    text = raw
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    match = re.search(r'<question>(.*?)</question>', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        text = _extract_question_from_raw(text)
    text = clean_model_response(text)
    return text


def _extract_question_from_raw(text: str) -> str:
    """Extract the actual question from verbose model output lacking <question> tags."""
    colon_split = re.split(r':\s*\n', text)
    if len(colon_split) > 1:
        candidate = colon_split[-1].strip()
        if len(candidate) > 20:
            return candidate

    sentences = re.split(r'(?<=[.?!])\s+', text)
    question_sentences = [s for s in sentences if s.strip().endswith('?')]
    if question_sentences:
        last_q_idx = len(sentences) - 1
        for idx in range(len(sentences) - 1, -1, -1):
            if sentences[idx].strip().endswith('?'):
                last_q_idx = idx
                break
        start_idx = last_q_idx
        for back in range(1, 3):
            check_idx = last_q_idx - back
            if check_idx >= 0 and re.search(r'\d', sentences[check_idx]):
                start_idx = check_idx
            else:
                break
        return ' '.join(sentences[start_idx:last_q_idx + 1]).strip()

    if len(sentences) > 2:
        return ' '.join(sentences[-2:]).strip()
    return text
