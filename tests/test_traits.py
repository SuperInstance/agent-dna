"""Tests for agent_dna.traits module."""

import pytest

from agent_dna.traits import (
    BUILTIN_TRAITS,
    Trait,
    TraitCategory,
    TraitRegistry,
    TRAIT_REGISTRY,
)


class TestTrait:
    def test_clamp_within_range(self):
        t = Trait(name="x", min_value=0.0, max_value=1.0)
        assert t.clamp(0.5) == 0.5

    def test_clamp_below(self):
        t = Trait(name="x", min_value=0.2, max_value=0.8)
        assert t.clamp(0.0) == 0.2

    def test_clamp_above(self):
        t = Trait(name="x", min_value=0.0, max_value=1.0)
        assert t.clamp(1.5) == 1.0

    def test_frozen(self):
        t = Trait(name="x")
        with pytest.raises(AttributeError):
            t.name = "y"  # type: ignore[misc]


class TestTraitRegistry:
    def test_register_and_get(self):
        reg = TraitRegistry()
        t = Trait(name="custom", min_value=0.0, max_value=10.0, default=5.0)
        reg.register(t)
        assert reg.get("custom") is t

    def test_get_missing(self):
        reg = TraitRegistry()
        assert reg.get("nonexistent") is None

    def test_contains(self):
        reg = TraitRegistry()
        reg.register(Trait(name="a"))
        assert "a" in reg
        assert "b" not in reg

    def test_len_and_iter(self):
        reg = TraitRegistry()
        reg.register(Trait(name="a"))
        reg.register(Trait(name="b"))
        assert len(reg) == 2
        names = {t.name for t in reg}
        assert names == {"a", "b"}

    def test_with_builtins(self):
        reg = TraitRegistry.with_builtins()
        assert len(reg) == len(BUILTIN_TRAITS)
        assert "persistence" in reg
        assert "creativity" in reg

    def test_default_registry_has_builtins(self):
        assert len(TRAIT_REGISTRY) == len(BUILTIN_TRAITS)


class TestBuiltinTraits:
    def test_all_have_names(self):
        for t in BUILTIN_TRAITS:
            assert t.name
            assert t.min_value < t.max_value
            assert t.min_value <= t.default <= t.max_value

    def test_categories_covered(self):
        cats = {t.category for t in BUILTIN_TRAITS}
        assert TraitCategory.BEHAVIOR in cats
        assert TraitCategory.COGNITIVE in cats
        assert TraitCategory.PERFORMANCE in cats
        assert TraitCategory.SOCIAL in cats
