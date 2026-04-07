"""
NanoGPT API client for scope data analysis.
Uses the OpenAI-compatible chat completions endpoint at nano-gpt.com.
"""

import json
import logging
import urllib.request
import urllib.error
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)

NANO_GPT_BASE_URL = "https://nano-gpt.com/api/v1"


class NanoGPTClient:
    """Lightweight NanoGPT API client using only stdlib (no requests dependency)."""

    # Default models shipped with the application
    DEFAULT_MODELS = [
        "anthropic/claude-opus-4.6",
    ]

    @staticmethod
    def load_model_list() -> list[str]:
        """Load the user's model list from QSettings, falling back to defaults."""
        from PySide6.QtCore import QSettings
        s = QSettings("TrioScope", "ParameterScope")
        saved = s.value("ai/model_list", None)
        if saved is not None and isinstance(saved, list) and len(saved) > 0:
            return list(saved)
        return list(NanoGPTClient.DEFAULT_MODELS)

    @staticmethod
    def save_model_list(models: list[str]):
        """Persist the user's model list to QSettings."""
        from PySide6.QtCore import QSettings
        s = QSettings("TrioScope", "ParameterScope")
        s.setValue("ai/model_list", models)

    def __init__(self, api_key: str = "", model: str = "openai/gpt-4.1-mini"):
        self.api_key = api_key
        self.model = model
        self._base_url = NANO_GPT_BASE_URL

    def set_api_key(self, key: str):
        self.api_key = key.strip()

    def set_model(self, model: str):
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], temperature: float = 0.3,
             max_tokens: int = 2048) -> str:
        """
        Synchronous chat completion. Returns the assistant's reply text.
        Raises RuntimeError on API errors.
        """
        if not self.api_key:
            raise RuntimeError("NanoGPT API key not configured. Set it in Settings.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error("NanoGPT API error %d: %s", e.code, error_body)
            raise RuntimeError(f"NanoGPT API error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

    def chat_async(self, messages: list[dict],
                   on_result: Callable[[str], None],
                   on_error: Callable[[str], None],
                   temperature: float = 0.3,
                   max_tokens: int = 2048):
        """
        Non-blocking chat completion. Runs in a background thread.
        Callbacks are called from the background thread — use Qt signals
        to forward results to the UI thread.
        """
        def _worker():
            try:
                result = self.chat(messages, temperature, max_tokens)
                on_result(result)
            except Exception as e:
                on_error(str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t

    def chat_stream(self, messages: list[dict],
                    on_chunk: Callable[[str], None],
                    on_done: Callable[[], None],
                    on_error: Callable[[str], None],
                    temperature: float = 0.3,
                    max_tokens: int = 2048):
        """
        Streaming chat completion in a background thread.
        on_chunk is called with each text delta as it arrives.
        """
        if not self.api_key:
            on_error("NanoGPT API key not configured. Set it in Settings.")
            return

        def _worker():
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._base_url}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue
                        payload_str = line[6:]  # strip "data: "
                        if payload_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                on_chunk(content)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                on_done()
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                on_error(f"API error {e.code}: {error_body}")
            except Exception as e:
                on_error(str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t
