from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .types import ApiName, ProviderName


class ModelCost(BaseModel):
    input_per_million: float = 0
    output_per_million: float = 0
    cache_read_per_million: float = 0
    cache_write_per_million: float = 0


class ModelSpec(BaseModel):
    id: str
    name: str
    api: ApiName
    provider: ProviderName
    base_url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    reasoning: bool = False
    input_modalities: list[Literal["text", "image", "file"]] = Field(
        default_factory=lambda: ["text"]
    )
    supports_tools: bool = False
    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: int | None = None
    max_output_tokens: int | None = None
    compat: dict[str, Any] = Field(default_factory=dict)


_model_registry: dict[str, ModelSpec] = {}


def register_model(model: ModelSpec) -> None:
    _model_registry[model.id] = model


def get_model(model_id: str) -> ModelSpec | None:
    return _model_registry.get(model_id)


def list_models() -> list[ModelSpec]:
    return list(_model_registry.values())


def clear_models() -> None:
    _model_registry.clear()
