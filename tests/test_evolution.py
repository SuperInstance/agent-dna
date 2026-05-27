"""Tests for agent_dna.evolution module."""

import pytest

from agent_dna.evolution import (
    SelectionStrategy,
    evolve,
    select,
)
from agent_dna.fitness import FitnessEvaluator, FitnessResult
from agent_dna.genome import AgentGenome


def _make_pop(n: int = 10):
    """Create a small population and its fitness results."""
    genomes = [AgentGenome.random() for _ in range(n)]
    ev = FitnessEvaluator()
    results = ev.evaluate_population(genomes)
    return genomes, results


class TestSelect:
    def test_tournament_returns_genome(self):
        genomes, results = _make_pop()
        ranked = [(g, r.composite) for g, r in zip(genomes, results)]
        chosen = select(ranked, strategy=SelectionStrategy.TOURNAMENT, tournament_k=3)
        assert isinstance(chosen, AgentGenome)

    def test_roulette_returns_genome(self):
        genomes, results = _make_pop()
        ranked = [(g, r.composite) for g, r in zip(genomes, results)]
        chosen = select(ranked, strategy=SelectionStrategy.ROULETTE)
        assert isinstance(chosen, AgentGenome)

    def test_rank_returns_genome(self):
        genomes, results = _make_pop()
        ranked = [(g, r.composite) for g, r in zip(genomes, results)]
        chosen = select(ranked, strategy=SelectionStrategy.RANK)
        assert isinstance(chosen, AgentGenome)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select([], strategy=SelectionStrategy.TOURNAMENT)


class TestEvolve:
    def test_returns_same_size(self):
        genomes, results = _make_pop(10)
        new_pop = evolve(genomes, results)
        assert len(new_pop) == 10

    def test_elites_preserved(self):
        genomes, results = _make_pop(10)
        # Find the best genome
        best_idx = max(range(len(results)), key=lambda i: results[i].composite)
        best_id = genomes[best_idx].genome_id

        new_pop = evolve(genomes, results, elite_count=1)
        new_ids = {g.genome_id for g in new_pop}
        assert best_id in new_ids

    def test_offspring_different_from_parents(self):
        genomes, results = _make_pop(10)
        new_pop = evolve(genomes, results, elite_count=0, mutation_rate=0.5)
        # With high mutation, offspring should differ from all originals
        orig_ids = {g.genome_id for g in genomes}
        new_ids = {g.genome_id for g in new_pop}
        assert len(orig_ids & new_ids) == 0  # all new IDs

    def test_empty_population(self):
        assert evolve([], []) == []

    def test_crossover_methods(self):
        genomes, results = _make_pop(6)
        for method in ("uniform", "blended"):
            new_pop = evolve(genomes, results, crossover_method=method)
            assert len(new_pop) == 6
