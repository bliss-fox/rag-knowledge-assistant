"""Unit tests for the local Ollama vision provider."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.ollama_vision_llm import OllamaVisionLLM, OllamaVisionLLMError


@dataclass
class MockLLMSettings:
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class MockVisionSettings:
    model: str = "llava"
    base_url: str = "http://localhost:11434"
    max_image_size: int = 2048


@dataclass
class MockSettings:
    llm: MockLLMSettings = field(default_factory=MockLLMSettings)
    vision_llm: MockVisionSettings = field(default_factory=MockVisionSettings)


def make_response(content: str = "A diagram", status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = content
    response.json.return_value = {
        "model": "llava",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 10,
        "eval_count": 6,
    }
    return response


def test_factory_configuration_and_base_url() -> None:
    provider = OllamaVisionLLM(MockSettings())
    assert provider.model == "llava"
    assert provider.base_url == "http://localhost:11434"


def test_chat_sends_ollama_images_payload() -> None:
    provider = OllamaVisionLLM(MockSettings())
    response = make_response()

    with patch("httpx.Client") as client:
        client.return_value.__enter__.return_value.post.return_value = response
        result = provider.chat_with_image("Describe", ImageInput(data=b"image-bytes"))

    call = client.return_value.__enter__.return_value.post.call_args
    assert call.args[0] == "http://localhost:11434/api/chat"
    payload = call.kwargs["json"]
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 8192
    assert payload["messages"][-1]["images"] == [base64.b64encode(b"image-bytes").decode("utf-8")]
    assert result.content == "A diagram"
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "total_tokens": 16,
    }


def test_connection_error_is_actionable() -> None:
    import httpx

    provider = OllamaVisionLLM(MockSettings())
    with patch("httpx.Client") as client:
        client.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(OllamaVisionLLMError, match="Ensure Ollama is running"):
            provider.chat_with_image("Describe", ImageInput(data=b"image-bytes"))
