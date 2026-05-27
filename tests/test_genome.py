"""Tests for agent_dna.genome module."""

import json

import pytest

from agent_dna.genome import AgentGenome, Gene
from agent_dna.traits import Trait, TraitRegistry, TRAIT_REGISTRY


class TestGene:
    def test_to_dict_roundtrip(self):
        g = Gene(trait="speed", value=0.7)
        d = g.to_dict()
        g2 = Gene.from_dict(d)
        assert g2.trait == "speed"
        assert g2.value == 0.7

    def test_from_dict_requires_fields(self):
        with pytest.raises(KeyError):
            Gene.from_dict({"trait": "x"})


class TestAgentGenomeCreation:
    def test_random_has_all_traits(self):
        g = AgentGenome.random()
        for trait in TRAIT_REGISTRY:
            assert trait.name in g.genes

    def test_random_values_in_range(self):
        g = AgentGenome.random()
        for trait in TRAIT_REGISTRY:
            val = g.get(trait.name)
            assert trait.min_value <= val <= trait.max_value

    def test_from_values(self):
        g = AgentGenome.from_values({"speed": 0.8, "caution": 0.3})
        assert g.get("speed") == 0.8
        assert g.get("caution") == 0.3
        assert g.get("nonexistent", 99) == 99

    def test_set_gene(self):
        g = AgentGenome()
        g.set("test_trait", 0.5)
        assert g.get("test_trait") == 0.5

    def test_default_has_unique_id(self):
        g1 = AgentGenome()
        g2 = AgentGenome()
        assert g1.genome_id != g2.genome_id


class TestMutation:
    def test_mutate_returns_new_genome(self):
        g = AgentGenome.from_values({"speed": 0.5})
        m = g.mutate(rate=1.0, sigma=0.1)
        assert m.genome_id != g.genome_id
        assert m.parent_ids == [g.genome_id]

    def test_mutate_with_zero_rate(self):
        g = AgentGenome.from_values({"speed": 0.5, "caution": 0.7})
        m = g.mutate(rate=0.0)
        assert m.get("speed") == 0.5
        assert m.get("caution") == 0.7

    def test_mutate_clamps(self):
        g = AgentGenome.from_values({"speed": 0.99})
        m = g.mutate(rate=1.0, sigma=5.0)  # huge sigma, should still clamp
        assert m.get("speed") <= 1.0
        assert m.get("speed") >= 0.0


class TestCrossover:
    def test_uniform_crossover(self):
        p1 = AgentGenome.from_values({"speed": 0.2, "caution": 0.8})
        p2 = AgentGenome.from_values({"speed": 0.9, "caution": 0.1})
        c1, c2 = p1.crossover(p2, method="uniform")
        assert len(c1.genes) == 2
        assert len(c2.genes) == 2
        assert c1.generation == 1
        assert set(c1.parent_ids) == {p1.genome_id, p2.genome_id}

    def test_blended_crossover(self):
        p1 = AgentGenome.from_values({"speed": 0.0})
        p2 = AgentGenome.from_values({"speed": 1.0})
        c1, c2 = p1.crossover(p2, method="blended", blend_alpha=0.5)
        # Children should be between parents (approximately, with noise)
        assert -0.1 <= c1.get("speed") <= 1.1  # noise tolerance
        assert -0.1 <= c2.get("speed") <= 1.1

    def test_crossover_different_traits(self):
        p1 = AgentGenome.from_values({"a": 0.5})
        p2 = AgentGenome.from_values({"b": 0.5})
        c1, c2 = p1.crossover(p2)
        assert "a" in c1.genes or "b" in c1.genes
        assert "a" in c2.genes or "b" in c2.genes


class TestSerialization:
    def test_to_dict_roundtrip(self):
        g = AgentGenome.from_values({"speed": 0.6, "caution": 0.4})
        d = g.to_dict()
        g2 = AgentGenome.from_dict(d)
        assert g2.get("speed") == 0.6
        assert g2.get("caution") == 0.4
        assert g2.generation == g.generation
        assert g2.genome_id == g.genome_id

    def test_json_roundtrip(self):
        g = AgentGenome.from_values({"speed": 0.3})
        j = g.to_json()
        g2 = AgentGenome.from_json(j)
        assert g2.get("speed") == 0.3
        # Verify it's valid JSON
        parsed = json.loads(j)
        assert "genes" in parsed

    def test_json_contains_expected_keys(self):
        g = AgentGenome.from_values({"speed": 0.5})
        d = g.to_dict()
        assert "genome_id" in d
        assert "generation" in d
        assert "parent_ids" in d
        assert "genes" in d


class TestDistance:
    def test_identical_genomes(self):
        g = AgentGenome.from_values({"speed": 0.5})
        assert g.distance(g) == 0.0

    def test_different_genomes(self):
        g1 = AgentGenome.from_values({"speed": 0.0})
        g2 = AgentGenome.from_values({"speed": 1.0})
        assert g1.distance(g2) > 0.0

    def test_empty_genomes(self):
        g1 = AgentGenome()
        g2 = AgentGenome()
        assert g1.distance(g2) == 0.0
