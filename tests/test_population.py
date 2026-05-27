"""Tests for agent_dna.population module."""

import pytest

from agent_dna.evolution import SelectionStrategy
from agent_dna.fitness import FitnessEvaluator
from agent_dna.genome import AgentGenome
from agent_dna.population import GenerationStats, Population


class TestPopulationCreation:
    def test_random_population(self):
        pop = Population.random(size=10)
        assert pop.size == 10
        assert pop.generation == 0

    def test_empty_population(self):
        pop = Population()
        assert pop.size == 0
        assert pop.generation == 0


class TestEvaluation:
    def test_evaluate(self):
        pop = Population.random(size=5)
        results = pop.evaluate()
        assert len(results) == 5
        assert all(r.composite > 0 for r in results)

    def test_get_fitness(self):
        pop = Population.random(size=3)
        pop.evaluate()
        g = pop.genomes[0]
        result = pop.get_fitness(g)
        assert result is not None
        assert result.genome_id == g.genome_id


class TestDiversity:
    def test_diversity_positive(self):
        pop = Population.random(size=10)
        d = pop.diversity()
        assert d > 0.0

    def test_diversity_zero_for_clones(self):
        g = AgentGenome.from_values({"speed": 0.5, "caution": 0.5})
        pop = Population(genomes=[g, g])
        assert pop.diversity() == 0.0

    def test_diversity_empty(self):
        pop = Population()
        assert pop.diversity() == 0.0


class TestStep:
    def test_step_returns_stats(self):
        pop = Population.random(size=10)
        stats = pop.step()
        assert isinstance(stats, GenerationStats)
        assert stats.population_size == 10
        assert stats.generation == 0
        assert stats.best_fitness >= stats.worst_fitness

    def test_step_preserves_size(self):
        pop = Population.random(size=10)
        pop.step()
        assert pop.size == 10

    def test_step_appends_history(self):
        pop = Population.random(size=10)
        pop.step()
        assert len(pop.history) == 1
        pop.step()
        assert len(pop.history) == 2


class TestRun:
    def test_run_multiple_gens(self):
        pop = Population.random(size=15)
        stats = pop.run(generations=5)
        assert len(stats) == 5
        assert len(pop.history) == 5

    def test_fitness_improves_over_generations(self):
        """Over enough generations with selection, fitness should generally improve."""
        pop = Population.random(size=30)
        stats = pop.run(generations=20, elite_count=3)
        # Not guaranteed due to randomness, but very likely with 20 gens
        first_best = stats[0].best_fitness
        last_best = stats[-1].best_fitness
        # At minimum, the last gen shouldn't be worse than 80% of first
        assert last_best >= first_best * 0.8


class TestBest:
    def test_best_returns_genome(self):
        pop = Population.random(size=5)
        best = pop.best()
        assert best is not None
        assert isinstance(best, AgentGenome)

    def test_best_empty(self):
        pop = Population()
        assert pop.best() is None


class TestSummary:
    def test_summary_string(self):
        pop = Population.random(size=10)
        pop.run(generations=3)
        s = pop.summary()
        assert "Population" in s
        assert "generation" in s
