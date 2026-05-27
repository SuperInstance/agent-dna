"""Tests for agent_dna.fitness module."""

import pytest

from agent_dna.fitness import (
    FitnessEvaluator,
    FitnessMetric,
    FitnessResult,
    adaptability_score,
    accuracy_score,
    efficiency_score,
    task_completion_score,
)
from agent_dna.genome import AgentGenome


class TestBuiltinScores:
    def test_task_completion_returns_float(self):
        g = AgentGenome.random()
        score = task_completion_score(g)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_efficiency_returns_float(self):
        g = AgentGenome.random()
        score = efficiency_score(g)
        assert 0.0 <= score <= 1.0

    def test_accuracy_returns_float(self):
        g = AgentGenome.random()
        score = accuracy_score(g)
        assert 0.0 <= score <= 1.0

    def test_adaptability_returns_float(self):
        g = AgentGenome.random()
        score = adaptability_score(g)
        assert 0.0 <= score <= 1.0


class TestFitnessResult:
    def test_composite_auto_computed(self):
        r = FitnessResult(genome_id="x", scores={"a": 0.5, "b": 0.7})
        assert r.composite == pytest.approx(0.6)

    def test_composite_empty_scores(self):
        r = FitnessResult(genome_id="x", scores={})
        assert r.composite == 0.0

    def test_composite_preserved_if_set(self):
        r = FitnessResult(genome_id="x", scores={"a": 0.5}, composite=0.9)
        assert r.composite == 0.9


class TestFitnessEvaluator:
    def test_evaluate_returns_result(self):
        ev = FitnessEvaluator()
        g = AgentGenome.random()
        result = ev.evaluate(g)
        assert result.genome_id == g.genome_id
        assert len(result.scores) == 4  # 4 builtin metrics
        assert 0.0 <= result.composite <= 1.0

    def test_evaluate_population(self):
        ev = FitnessEvaluator()
        genomes = [AgentGenome.random() for _ in range(5)]
        results = ev.evaluate_population(genomes)
        assert len(results) == 5
        assert all(isinstance(r, FitnessResult) for r in results)

    def test_custom_weights(self):
        ev = FitnessEvaluator(
            weights={FitnessMetric.TASK_COMPLETION: 1.0, FitnessMetric.EFFICIENCY: 0.0}
        )
        g = AgentGenome.random()
        result = ev.evaluate(g)
        # With zero weight on efficiency, composite should equal task_completion
        tc = result.scores[FitnessMetric.TASK_COMPLETION]
        assert result.composite == pytest.approx(tc)

    def test_custom_metric(self):
        ev = FitnessEvaluator(
            metrics={"custom": lambda g: g.get("speed", 0.0)}
        )
        g = AgentGenome.from_values({"speed": 0.8})
        result = ev.evaluate(g)
        assert result.scores["custom"] == 0.8
