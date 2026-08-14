"""Validated, versioned prompt registry."""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VersionedPrompt:
    prompt_id: str
    version: str
    description: str
    template: str
    required_variables: tuple[str, ...]
    sha256: str

    def render(self, **variables: Any) -> str:
        missing = [name for name in self.required_variables if name not in variables]
        if missing:
            raise ValueError(f"Missing prompt variables: {', '.join(missing)}")
        return self.template.format(**variables)


class PromptRegistry:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._prompts: dict[str, VersionedPrompt] = {}
        self.reload()

    def reload(self) -> None:
        prompts: dict[str, VersionedPrompt] = {}
        for path in sorted((*self.directory.glob("*.yaml"), *self.directory.glob("*.yml"))):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            prompt = self._parse(data, path)
            if prompt.prompt_id in prompts:
                raise ValueError(f"Duplicate prompt id: {prompt.prompt_id}")
            prompts[prompt.prompt_id] = prompt
        if not prompts:
            raise ValueError(f"No versioned prompts found in {self.directory}")
        self._prompts = prompts

    def get(self, prompt_id: str) -> VersionedPrompt:
        try:
            return self._prompts[prompt_id]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt: {prompt_id}") from exc

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"id": item.prompt_id, "version": item.version, "description": item.description,
             "required_variables": list(item.required_variables), "sha256": item.sha256}
            for item in self._prompts.values()
        ]

    @staticmethod
    def _parse(data: dict[str, Any], path: Path) -> VersionedPrompt:
        required = {"id", "version", "description", "template", "required_variables"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        required_variables = tuple(str(item) for item in data["required_variables"])
        template = str(data["template"])
        fields = {
            name for _, name, _, _ in string.Formatter().parse(template) if name
        }
        if fields != set(required_variables):
            raise ValueError(
                f"{path}: template variables {sorted(fields)} do not match required_variables {sorted(required_variables)}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return VersionedPrompt(
            prompt_id=str(data["id"]), version=str(data["version"]),
            description=str(data["description"]), template=template,
            required_variables=required_variables, sha256=digest,
        )
