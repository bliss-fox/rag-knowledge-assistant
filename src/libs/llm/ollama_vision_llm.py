"""Ollama Vision LLM implementation for local multimodal inference."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class OllamaVisionLLMError(RuntimeError):
    """Raised when an Ollama vision request fails."""


class OllamaVisionLLM(BaseVisionLLM):
    """Vision provider backed by Ollama's local ``/api/chat`` endpoint."""

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 180.0
    DEFAULT_CONTEXT_WINDOW = 8192

    def __init__(
        self,
        settings: Any,
        base_url: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        vision_settings = getattr(settings, "vision_llm", None)
        if vision_settings is None:
            raise ValueError("Missing required configuration: settings.vision_llm")

        self.model = vision_settings.model
        self.base_url = (
            base_url
            or getattr(vision_settings, "base_url", None)
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.default_temperature = getattr(settings.llm, "temperature", 0.0)
        self.default_max_tokens = getattr(settings.llm, "max_tokens", 4096)
        self.max_image_size = getattr(vision_settings, "max_image_size", 2048)
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.context_window = kwargs.pop("num_ctx", self.DEFAULT_CONTEXT_WINDOW)
        self._extra_config = kwargs

    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: list[Message] | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.validate_text(text)
        self.validate_image(image)

        api_messages = []
        if messages:
            api_messages.extend({"role": item.role, "content": item.content} for item in messages)
        api_messages.append(
            {
                "role": "user",
                "content": text,
                "images": [self._get_image_base64(image)],
            }
        )

        response_data = self._call_api(
            messages=api_messages,
            temperature=kwargs.get("temperature", self.default_temperature),
            max_tokens=kwargs.get("max_tokens", self.default_max_tokens),
            context_window=kwargs.get("num_ctx", self.context_window),
        )

        try:
            content = response_data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaVisionLLMError(
                "[Ollama Vision] Unexpected response format: missing message.content"
            ) from exc

        usage = None
        if "eval_count" in response_data or "prompt_eval_count" in response_data:
            prompt_tokens = response_data.get("prompt_eval_count", 0)
            completion_tokens = response_data.get("eval_count", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        return ChatResponse(
            content=content,
            model=response_data.get("model", self.model),
            usage=usage,
            raw_response=response_data,
        )

    @staticmethod
    def _get_image_base64(image: ImageInput) -> str:
        try:
            if image.base64:
                return image.base64
            if image.data is not None:
                return base64.b64encode(image.data).decode("utf-8")
            if image.path is not None:
                return base64.b64encode(Path(image.path).read_bytes()).decode("utf-8")
        except OSError as exc:
            raise OllamaVisionLLMError(f"[Ollama Vision] Failed to read image: {exc}") from exc
        raise OllamaVisionLLMError("[Ollama Vision] Image has no data source")

    def _call_api(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        context_window: int,
    ) -> dict[str, Any]:
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": context_window,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                if response.status_code != 200:
                    try:
                        detail = response.json().get("error", response.text[:200])
                    except Exception:
                        detail = response.text[:200] or "Unknown error"
                    raise OllamaVisionLLMError(
                        f"[Ollama Vision] API error (HTTP {response.status_code}): {detail}"
                    )
                return response.json()
        except httpx.TimeoutException as exc:
            raise OllamaVisionLLMError(
                f"[Ollama Vision] Request timed out after {self.timeout} seconds"
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaVisionLLMError(
                "[Ollama Vision] Connection failed. Ensure Ollama is running locally."
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaVisionLLMError(
                f"[Ollama Vision] Request failed: {type(exc).__name__}"
            ) from exc
