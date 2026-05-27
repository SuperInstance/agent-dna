"""Population management with diversity tracking and generational evolution."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .evolution import SelectionStrategy, evolve
from .fitness import FitnessEvaluator, FitnessResult
from .genome import AgentGenome
from .traits import TraitRegistry, TRAIT_REGISTRY


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    population_size: int
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    std_fitness: float
    diversity: float  # average pairwise distance
    best_genome_id: str


@dataclass
class Population:
    """Manages a population of agent genomes across generations.

    Provides methods for initialization, evaluation, evolution, and tracking
    diversity and fitness statistics over time.
    """
    genomes: List[AgentGenome] = field(default_factory=list)
    evaluator: FitnessEvaluator = field(default_factory=FitnessEvaluator)
    registry: TraitRegistry = field(default_factory=lambda: TRAIT_REGISTRY)
    history: List[GenerationStats] = field(default_factory=list)
    _fitness_cache: Dict[str, FitnessResult] = field(default_factory=dict)

    # --- Initialization ---

    @classmethod
    def random(
        cls,
        size: int = 20,
        registry: TraitRegistry = TRAIT_REGISTRY,
        evaluator: Optional[FitnessEvaluator] = None,
    ) -> Population:
        """Create a random population of the given size."""
        genomes = [AgentGenome.random(registry=registry) for _ in range(size)]
        return cls(
            genomes=genomes,
            registry=registry,
            evaluator=evaluator or FitnessEvaluator(),
        )

    # --- Properties ---

    @property
    def size(self) -> int:
        return len(self.genomes)

    @property
    def generation(self) -> int:
        if not self.genomes:
            return 0
        return max(g.generation for g in self.genomes)

    # --- Evaluation ---

    def evaluate(self) -> List[FitnessResult]:
        """Evaluate all genomes and cache results."""
        results = self.evaluator.evaluate_population(self.genomes)
        self._fitness_cache = {r.genome_id: r for r in results}
        return results

    def get_fitness(self, genome: AgentGenome) -> Optional[FitnessResult]:
        """Get cached fitness for a genome, if evaluated."""
        return self._fitness_cache.get(genome.genome_id)

    # --- Diversity ---

    def diversity(self) -> float:
        """Average pairwise Euclidean distance between all genomes.

        O(n²) but fine for typical population sizes (< 1000).
        """
        n = len(self.genomes)
        if n < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += self.genomes[i].distance(self.genomes[j])
                count += 1
        return total_dist / count

    # --- Evolution ---

    def step(
        self,
        elite_count: int = 2,
        mutation_rate: float = 0.15,
        mutation_sigma: float = 0.12,
        crossover_method: str = "uniform",
        strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT,
        tournament_k: int = 3,
    ) -> GenerationStats:
        """Run one generation: evaluate, record stats, evolve.

        Returns:
            GenerationStats for the current generation (before evolution).
        """
        results = self.evaluate()

        # Compute stats
        scores = [r.composite for r in results]
        best_idx = max(range(len(results)), key=lambda i: scores[i])
        div = self.diversity()

        stats = GenerationStats(
            generation=self.generation,
            population_size=self.size,
            best_fitness=max(scores) if scores else 0.0,
            worst_fitness=min(scores) if scores else 0.0,
            mean_fitness=statistics.mean(scores) if scores else 0.0,
            std_fitness=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            diversity=round(div, 6),
            best_genome_id=self.genomes[best_idx].genome_id if self.genomes else "",
        )
        self.history.append(stats)

        # Evolve
        self.genomes = evolve(
            self.genomes,
            results,
            elite_count=elite_count,
            mutation_rate=mutation_rate,
            mutation_sigma=mutation_sigma,
            crossover_method=crossover_method,
            strategy=strategy,
            tournament_k=tournament_k,
            registry=self.registry,
        )

        return stats

    def run(
        self,
        generations: int = 10,
        elite_count: int = 2,
        mutation_rate: float = 0.15,
        mutation_sigma: float = 0.12,
        crossover_method: str = "uniform",
        strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT,
        tournament_k: int = 3,
    ) -> List[GenerationStats]:
        """Run multiple generations of evolution.

        Returns:
            List of GenerationStats, one per generation.
        """
        all_stats: List[GenerationStats] = []
        for _ in range(generations):
            stats = self.step(
                elite_count=elite_count,
                mutation_rate=mutation_rate,
                mutation_sigma=mutation_sigma,
                crossover_method=crossover_method,
                strategy=strategy,
                tournament_k=tournament_k,
            )
            all_stats.append(stats)
        return all_stats

    # --- Best genome ---

    def best(self) -> Optional[AgentGenome]:
        """Return the genome with the highest cached fitness."""
        if not self._fitness_cache:
            self.evaluate()
        if not self._fitness_cache:
            return None
        best_id = max(self._fitness_cache, key=lambda k: self._fitness_cache[k].composite)
        for g in self.genomes:
            if g.genome_id == best_id:
                return g
        return None

    # --- Summary ---

    def summary(self) -> str:
        """Human-readable summary of the population and its history."""
        lines = [
            f"Population(size={self.size}, generation={self.generation})",
        ]
        if self.history:
            last = self.history[-1]
            lines.append(f"  Last gen: best={last.best_fitness:.4f}, "
                         f"mean={last.mean_fitness:.4f}, "
                         f"diversity={last.diversity:.4f}")
        if len(self.history) > 1:
            first, last = self.history[0], self.history[-1]
            delta = last.best_fitness - first.best_fitness
            lines.append(f"  Improvement over {len(self.history)} gens: {delta:+.4f}")
        return "\n".join(lines)
