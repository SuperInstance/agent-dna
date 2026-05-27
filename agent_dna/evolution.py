"""Evolutionary operators: selection, crossover, mutation, and the evolve() loop."""

from __future__ import annotations

import random
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .fitness import FitnessEvaluator, FitnessResult
from .genome import AgentGenome
from .traits import TraitRegistry, TRAIT_REGISTRY


class SelectionStrategy(str, Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


def _tournament_select(
    ranked: List[Tuple[AgentGenome, float]],
    k: int = 3,
) -> AgentGenome:
    """Select one genome via tournament selection."""
    contestants = random.sample(ranked, min(k, len(ranked)))
    return max(contestants, key=lambda x: x[1])[0]


def _roulette_select(ranked: List[Tuple[AgentGenome, float]]) -> AgentGenome:
    """Select one genome via fitness-proportionate (roulette wheel) selection."""
    total = sum(f for _, f in ranked)
    if total <= 0:
        return random.choice(ranked)[0]
    r = random.uniform(0, total)
    cumulative = 0.0
    for genome, fitness in ranked:
        cumulative += fitness
        if cumulative >= r:
            return genome
    return ranked[-1][0]


def _rank_select(ranked: List[Tuple[AgentGenome, float]]) -> AgentGenome:
    """Select one genome via rank-based selection."""
    n = len(ranked)
    # ranked is assumed sorted best-first; higher rank = higher chance
    total_rank = n * (n + 1) / 2
    r = random.uniform(0, total_rank)
    cumulative = 0.0
    for i, (genome, _) in enumerate(reversed(ranked)):
        cumulative += (i + 1)
        if cumulative >= r:
            return genome
    return ranked[-1][0]


def select(
    ranked: List[Tuple[AgentGenome, float]],
    strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT,
    tournament_k: int = 3,
) -> AgentGenome:
    """Select a single genome using the given strategy."""
    if not ranked:
        raise ValueError("Cannot select from empty population")
    if strategy == SelectionStrategy.TOURNAMENT:
        return _tournament_select(ranked, tournament_k)
    elif strategy == SelectionStrategy.ROULETTE:
        return _roulette_select(ranked)
    elif strategy == SelectionStrategy.RANK:
        return _rank_select(ranked)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evolve(
    population: Sequence[AgentGenome],
    fitness_results: Sequence[FitnessResult],
    *,
    elite_count: int = 2,
    offspring_size: Optional[int] = None,
    mutation_rate: float = 0.15,
    mutation_sigma: float = 0.12,
    crossover_method: str = "uniform",
    strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT,
    tournament_k: int = 3,
    registry: TraitRegistry = TRAIT_REGISTRY,
) -> List[AgentGenome]:
    """Run one generation of evolution.

    Args:
        population: Current population of genomes.
        fitness_results: Pre-computed fitness results (must align with population).
        elite_count: Number of top genomes to carry forward unchanged.
        offspring_size: How many children to produce (defaults to len(population) - elite_count).
        mutation_rate: Per-gene mutation probability.
        mutation_sigma: Gaussian mutation std-dev.
        crossover_method: "uniform" or "blended".
        strategy: Parent selection strategy.
        tournament_k: Tournament size (only for tournament selection).
        registry: Trait registry for clamping.

    Returns:
        New population of the same size as the input.
    """
    if not population:
        return []

    # Pair genomes with fitness and sort descending
    pairs = list(zip(population, fitness_results))
    pairs.sort(key=lambda x: x[1].composite, reverse=True)
    ranked: List[Tuple[AgentGenome, float]] = [
        (g, r.composite) for g, r in pairs
    ]

    pop_size = len(population)
    if offspring_size is None:
        offspring_size = pop_size - elite_count

    # Elitism: carry forward top genomes
    elites = [g for g, _ in ranked[:elite_count]]

    # Selection pool
    new_pop = list(elites)

    while len(new_pop) < pop_size:
        parent1 = select(ranked, strategy, tournament_k)
        parent2 = select(ranked, strategy, tournament_k)

        child1, child2 = parent1.crossover(
            parent2, method=crossover_method, registry=registry
        )
        child1 = child1.mutate(rate=mutation_rate, sigma=mutation_sigma, registry=registry)
        child2 = child2.mutate(rate=mutation_rate, sigma=mutation_sigma, registry=registry)

        new_pop.append(child1)
        if len(new_pop) < pop_size:
            new_pop.append(child2)

    return new_pop[:pop_size]
