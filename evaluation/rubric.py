"""Agentic Rubric 加载与校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_RUBRIC_PATH = Path(__file__).with_name("rubric.yaml")


@dataclass(frozen=True)
class RubricDimension:
    dimension_id: str
    name: str
    weight: float
    description: str


@dataclass(frozen=True)
class Rubric:
    version: str
    dimensions: tuple[RubricDimension, ...]

    @property
    def total_weight(self) -> float:
        return sum(item.weight for item in self.dimensions)


def load_rubric(path: str | Path = DEFAULT_RUBRIC_PATH) -> Rubric:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("rubric root must be a mapping")
    raw_dimensions = data.get("dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise ValueError("rubric dimensions must be a non-empty list")
    dimensions: list[RubricDimension] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_dimensions):
        if not isinstance(raw, dict):
            raise ValueError(f"dimension {index} must be a mapping")
        dimension_id = raw.get("id")
        name = raw.get("name")
        description = raw.get("description")
        weight = raw.get("weight")
        if not all(isinstance(value, str) and value.strip() for value in (dimension_id, name, description)):
            raise ValueError(f"dimension {index} has invalid text fields")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            raise ValueError(f"dimension {dimension_id} has invalid weight")
        if dimension_id in seen:
            raise ValueError(f"duplicate rubric dimension: {dimension_id}")
        seen.add(dimension_id)
        dimensions.append(RubricDimension(dimension_id, name, float(weight), description))
    rubric = Rubric(str(data.get("version") or "unknown"), tuple(dimensions))
    if abs(rubric.total_weight - 100.0) > 1e-6:
        raise ValueError(f"rubric weights must sum to 100, got {rubric.total_weight}")
    return rubric
