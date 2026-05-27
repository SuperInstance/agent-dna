"""Agent genome representation with mutation and crossover support."""

from __future__ import annotations

import copy
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .traits import TraitRegistry, TRAIT_REGISTRY


@dataclass
class Gene:
    """A single gene: a trait name paired with a float value."""
    trait: str
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return {"trait": self.trait, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Gene:
        return cls(trait=data["trait"], value=data["value"])


@dataclass
class AgentGenome:
    """An agent's genome — a collection of genes mapped to trait values.

    Supports mutation (gaussian perturbation), crossover (uniform / blended),
    and JSON serialization.
    """
    genes: Dict[str, Gene] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # --- Factory helpers ---

    @classmethod
    def random(
        cls,
        registry: TraitRegistry = TRAIT_REGISTRY,
        generation: int = 0,
    ) -> AgentGenome:
        """Create a genome with random values for all registered traits."""
        genes: Dict[str, Gene] = {}
        for trait in registry:
            val = random.uniform(trait.min_value, trait.max_value)
            genes[trait.name] = Gene(trait=trait.name, value=round(val, 6))
        return cls(genes=genes, generation=generation)

    @classmethod
    def from_values(
        cls,
        values: Dict[str, float],
        generation: int = 0,
    ) -> AgentGenome:
        """Create a genome from an explicit {trait_name: value} mapping."""
        genes = {name: Gene(trait=name, value=val) for name, val in values.items()}
        return cls(genes=genes, generation=generation)

    # --- Access ---

    def get(self, trait_name: str, default: float = 0.0) -> float:
        gene = self.genes.get(trait_name)
        return gene.value if gene else default

    def set(self, trait_name: str, value: float) -> None:
        self.genes[trait_name] = Gene(trait=trait_name, value=value)

    # --- Mutation ---

    def mutate(
        self,
        rate: float = 0.1,
        sigma: float = 0.15,
        registry: TraitRegistry = TRAIT_REGISTRY,
    ) -> AgentGenome:
        """Return a new genome with gaussian mutation applied.

        Args:
            rate: Probability of mutating each gene.
            sigma: Std-dev of the gaussian perturbation.
            registry: Trait registry for clamping.

        Returns:
            A new AgentGenome (self is unchanged).
        """
        new_genes: Dict[str, Gene] = {}
        for name, gene in self.genes.items():
            if random.random() < rate:
                trait = registry.get(name)
                val = gene.value + random.gauss(0, sigma)
                if trait:
                    val = trait.clamp(val)
                new_genes[name] = Gene(trait=name, value=round(val, 6))
            else:
                new_genes[name] = Gene(trait=name, value=gene.value)
        return AgentGenome(
            genes=new_genes,
            generation=self.generation,
            parent_ids=[self.genome_id],
        )

    # --- Crossover ---

    def crossover(
        self,
        other: AgentGenome,
        method: str = "uniform",
        blend_alpha: float = 0.5,
        registry: TraitRegistry = TRAIT_REGISTRY,
    ) -> Tuple[AgentGenome, AgentGenome]:
        """Produce two offspring via crossover with *other*.

        Args:
            method: "uniform" (coin-flip per gene) or "blended" (weighted avg).
            blend_alpha: Interpolation weight for blended crossover (0→self, 1→other).

        Returns:
            Two child AgentGenomes.
        """
        all_traits = set(self.genes) | set(other.genes)
        child1_genes: Dict[str, Gene] = {}
        child2_genes: Dict[str, Gene] = {}

        for name in all_traits:
            v1 = self.get(name)
            v2 = other.get(name)

            if method == "blended":
                c1 = v1 * (1 - blend_alpha) + v2 * blend_alpha
                c2 = v2 * (1 - blend_alpha) + v1 * blend_alpha
                # Add small random deviation so children aren't identical
                c1 += random.gauss(0, 0.02)
                c2 += random.gauss(0, 0.02)
                trait = registry.get(name)
                if trait:
                    c1 = trait.clamp(c1)
                    c2 = trait.clamp(c2)
                child1_genes[name] = Gene(trait=name, value=round(c1, 6))
                child2_genes[name] = Gene(trait=name, value=round(c2, 6))
            else:  # uniform
                if random.random() < 0.5:
                    child1_genes[name] = Gene(trait=name, value=v1)
                    child2_genes[name] = Gene(trait=name, value=v2)
                else:
                    child1_genes[name] = Gene(trait=name, value=v2)
                    child2_genes[name] = Gene(trait=name, value=v1)

        parent_ids = [self.genome_id, other.genome_id]
        next_gen = max(self.generation, other.generation) + 1

        child1 = AgentGenome(genes=child1_genes, generation=next_gen, parent_ids=parent_ids)
        child2 = AgentGenome(genes=child2_genes, generation=next_gen, parent_ids=parent_ids)
        return child1, child2

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "genes": {name: gene.to_dict() for name, gene in self.genes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentGenome:
        genes = {name: Gene.from_dict(g) for name, g in data.get("genes", {}).items()}
        return cls(
            genes=genes,
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            genome_id=data.get("genome_id", uuid.uuid4().hex[:12]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> AgentGenome:
        return cls.from_dict(json.loads(json_str))

    # --- Distance ---

    def distance(self, other: AgentGenome) -> float:
        """Euclidean distance between two genomes' gene values."""
        all_traits = set(self.genes) | set(other.genes)
        if not all_traits:
            return 0.0
        sq_sum = 0.0
        for name in all_traits:
            diff = self.get(name) - other.get(name)
            sq_sum += diff * diff
        return math.sqrt(sq_sum / len(all_traits))

    def __repr__(self) -> str:
        short = self.genome_id[:8]
        return f"AgentGenome(id={short!r}, gen={self.generation}, genes={len(self.genes)})"
