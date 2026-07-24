from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """
    The output of the Feature Engineering Pipeline for one symbol at one
    point in time. `values` holds every computed feature by name; `is_complete`
    is False if any value is NaN (i.e. insufficient history) — later stages
    should not trade off an incomplete vector.
    """

    symbol: str
    epoch: int
    values: dict[str, float] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return all(v == v for v in self.values.values())  # v == v is False only for NaN

    def get(self, name: str) -> float:
        return self.values[name]
