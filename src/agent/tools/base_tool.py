"""Abstract base class for ReAct agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base for all agent tools.

    Subclasses must define name, description, and parameters_schema,
    and implement run(**kwargs) returning a natural-language string
    that becomes the Observation in the ReAct loop.
    """

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return an observation string."""

    def __call__(self, **kwargs: Any) -> str:
        return self.run(**kwargs)
