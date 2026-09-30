"""OpenAI-compatible chat client.

The default base URL is Ollama at ``http://localhost:11434/v1``. Point
``LLM_BASE_URL`` at a vLLM server to switch runtimes without changing
retrieval or prompts.
"""

import time

import httpx

from core.errors import ModelUnavailableError
from generation.provider import GenerationResult


class OpenAICompatibleProvider:
    """POST ``{base_url}/chat/completions`` and read the first choice."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "ollama",
        timeout_seconds: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model_name = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client = client

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self._base_url}/chat/completions"
        started = time.perf_counter()
        try:
            response = self._post(url, payload, headers)
            response.raise_for_status()
            body = response.json()
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                f"LLM request to {self._base_url} failed: {exc}"
            ) from exc
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError("LLM response did not contain a message.") from exc
        usage = body.get("usage") or {}
        return GenerationResult(
            text=text or "",
            model=str(body.get("model") or self.model_name),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _post(self, url: str, payload: dict, headers: dict) -> httpx.Response:
        if self._client is not None:
            return self._client.post(url, json=payload, headers=headers)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                return client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                f"Could not reach the OpenAI-compatible server at {self._base_url}. "
                "Start Ollama or set LLM_BASE_URL to a running vLLM server. "
                f"Underlying error: {exc}"
            ) from exc
