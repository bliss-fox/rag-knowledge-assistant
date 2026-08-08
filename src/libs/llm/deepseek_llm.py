"""DeepSeek LLM implementation.

This module provides the DeepSeek LLM implementation that works with
DeepSeek's API. DeepSeek uses an OpenAI-compatible API format but with
its own endpoint and authentication.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class DeepSeekLLMError(RuntimeError):
    """Raised when DeepSeek API call fails."""


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM provider implementation.

    This class implements the BaseLLM interface for DeepSeek's chat API.
    DeepSeek provides an OpenAI-compatible API with its own endpoint.

    Attributes:
        api_key: The API key for authentication.
        base_url: The base URL for the API.
        model: The model identifier to use.
        default_temperature: Default temperature for generation.
        default_max_tokens: Default max tokens for generation.

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = DeepSeekLLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        settings: Any,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DeepSeek LLM provider.

        Args:
            settings: Application settings containing LLM configuration.
            api_key: Optional API key override (falls back to env var DEEPSEEK_API_KEY).
            base_url: Optional base URL override.
            **kwargs: Additional configuration overrides.

        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens

        # Read local credentials without placing secrets in tracked YAML files or
        # mutating process-wide environment variables during tests.
        repo_root = Path(__file__).resolve().parents[3]
        dotenv_api_key = None
        if settings.__class__.__module__ == "src.core.settings":
            dotenv_api_key = dotenv_values(repo_root / ".env").get("DEEPSEEK_API_KEY")

        # API key: explicit > environment > local .env file
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or dotenv_api_key
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Base URL: explicit > settings > default
        self.base_url = base_url or getattr(settings.llm, "base_url", None) or self.DEFAULT_BASE_URL

        # Store any additional kwargs for future use
        self._extra_config = kwargs

    def chat(
        self,
        messages: list[Message],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion using DeepSeek API.

        Args:
            messages: List of conversation messages.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Override parameters (temperature, max_tokens, etc.).

        Returns:
            ChatResponse with generated content and metadata.

        Raises:
            ValueError: If messages are invalid.
            DeepSeekLLMError: If API call fails.
        """
        # Validate input
        self.validate_messages(messages)

        # Prepare request parameters
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)

        # Convert messages to API format
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Make API call
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Parse response
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")

            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise DeepSeekLLMError(f"[DeepSeek] Unexpected response format: missing key {e}") from e
        except Exception as e:
            if isinstance(e, DeepSeekLLMError):
                raise
            raise DeepSeekLLMError(f"[DeepSeek] API call failed: {type(e).__name__}: {e}") from e

    def stream_chat(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream tokens via DeepSeek SSE (OpenAI-compatible)."""
        import json as _json

        import httpx

        self.validate_messages(messages)
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.default_temperature),
            "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
            "stream": True,
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content") or ""
                        if delta:
                            yield delta
                    except (ValueError, KeyError, IndexError):
                        continue

    def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Make the actual API call to DeepSeek.

        This method is separated to allow easy mocking in tests.

        Args:
            messages: Messages in API format.
            model: Model identifier.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Raw API response as dictionary.

        Raises:
            DeepSeekLLMError: If the API call fails.
        """
        import httpx

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)

                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise DeepSeekLLMError(
                        f"[DeepSeek] API error (HTTP {response.status_code}): {error_detail}"
                    )

                return response.json()
        except httpx.TimeoutException as e:
            raise DeepSeekLLMError("[DeepSeek] Request timed out after 60 seconds") from e
        except httpx.RequestError as e:
            raise DeepSeekLLMError(f"[DeepSeek] Connection failed: {type(e).__name__}: {e}") from e

    def _parse_error_response(self, response: Any) -> str:
        """Parse error details from API response.

        Args:
            response: The HTTP response object.

        Returns:
            Human-readable error message.
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    return error.get("message", str(error))
                return str(error)
            return response.text
        except Exception:
            return response.text or "Unknown error"
