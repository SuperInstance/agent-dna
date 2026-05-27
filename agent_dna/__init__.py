"""agent-dna — Genetic encoding for agent capabilities and behavior traits."""

from .genome import AgentGenome, Gene
from .traits import Trait, TraitRegistry, TRAIT_REGISTRY
from .fitness import FitnessEvaluator, FitnessResult
from .evolution import evolve, SelectionStrategy
from .population import Population

__version__ = "0.1.0"
__all__ = [
    "AgentGenome",
    "Gene",
    "Trait",
    "TraitRegistry",
    "TRAIT_REGISTRY",
    "FitnessEvaluator",
    "FitnessResult",
    "evolve",
    "SelectionStrategy",
    "Population",
]
