"""Trait definitions for agent genomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, Optional


class TraitCategory(str, Enum):
    """Categories of agent traits."""
    BEHAVIOR = "behavior"
    COGNITIVE = "cognitive"
    PERFORMANCE = "performance"
    SOCIAL = "social"


@dataclass(frozen=True)
class Trait:
    """Defines a single trait that can appear in an agent genome.

    Attributes:
        name: Unique identifier (e.g. "persistence").
        min_value: Lower bound for the gene value.
        max_value: Upper bound for the gene value.
        default: Default value when not specified.
        category: Which category this trait belongs to.
        description: Human-readable explanation.
    """
    name: str
    min_value: float = 0.0
    max_value: float = 1.0
    default: float = 0.5
    category: TraitCategory = TraitCategory.BEHAVIOR
    description: str = ""

    def clamp(self, value: float) -> float:
        """Clamp a value to this trait's valid range."""
        return max(self.min_value, min(self.max_value, value))


# --- Built-in trait definitions ---

BUILTIN_TRAITS: list[Trait] = [
    Trait(
        name="persistence",
        min_value=0.0, max_value=1.0, default=0.5,
        category=TraitCategory.BEHAVIOR,
        description="How long an agent continues before giving up on a task.",
    ),
    Trait(
        name="creativity",
        min_value=0.0, max_value=1.0, default=0.5,
        category=TraitCategory.COGNITIVE,
        description="Tendency to explore novel solutions vs. following known paths.",
    ),
    Trait(
        name="caution",
        min_value=0.0, max_value=1.0, default=0.5,
        category=TraitCategory.BEHAVIOR,
        description="Risk aversion — how carefully an agent validates before acting.",
    ),
    Trait(
        name="speed",
        min_value=0.0, max_value=1.0, default=0.5,
        category=TraitCategory.PERFORMANCE,
        description="Preference for fast execution over thorough analysis.",
    ),
    Trait(
        name="verbosity",
        min_value=0.0, max_value=1.0, default=0.5,
        category=TraitCategory.SOCIAL,
        description="How much detail the agent includes in responses.",
    ),
    Trait(
        name="adaptability",
        min_value=0.0, max_value=1.0, default=0.5,
        category=TraitCategory.COGNITIVE,
        description="How quickly an agent adjusts to changing requirements.",
    ),
    Trait(
        name="thoroughness",
        min_value=0.0, max_value=1.0, default=0.5,
        category=TraitCategory.PERFORMANCE,
        description="Depth of analysis before producing output.",
    ),
    Trait(
        name="cooperativeness",
        min_value=0.0, max_value=1.0, default=0.5,
        category=TraitCategory.SOCIAL,
        description="Willingness to defer to or collaborate with other agents.",
    ),
]


@dataclass
class TraitRegistry:
    """Registry of known traits. Acts as the gene blueprint for genomes."""
    _traits: Dict[str, Trait] = field(default_factory=dict)

    def register(self, trait: Trait) -> None:
        self._traits[trait.name] = trait

    def get(self, name: str) -> Optional[Trait]:
        return self._traits.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._traits

    def __iter__(self) -> Iterator[Trait]:
        return iter(self._traits.values())

    def __len__(self) -> int:
        return len(self._traits)

    @classmethod
    def with_builtins(cls) -> TraitRegistry:
        """Create a registry pre-loaded with the built-in traits."""
        registry = cls()
        for trait in BUILTIN_TRAITS:
            registry.register(trait)
        return registry


# Module-level default registry
TRAIT_REGISTRY = TraitRegistry.with_builtins()
