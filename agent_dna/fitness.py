"""Fitness evaluation functions for agent genomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

from .genome import AgentGenome
from .traits import TRAIT_REGISTRY


class FitnessMetric(str, Enum):
    TASK_COMPLETION = "task_completion"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    ADAPTABILITY = "adaptability"


@dataclass
class FitnessResult:
    """Result of evaluating a genome's fitness."""
    genome_id: str
    scores: Dict[str, float]
    composite: float = 0.0

    def __post_init__(self) -> None:
        if not self.composite and self.scores:
            self.composite = sum(self.scores.values()) / len(self.scores)


# --- Built-in evaluation functions ---

def task_completion_score(genome: AgentGenome) -> float:
    """Score based on persistence and thoroughness — agents that stick with tasks
    and are thorough complete more tasks."""
    persistence = genome.get("persistence", 0.5)
    thoroughness = genome.get("thoroughness", 0.5)
    speed = genome.get("speed", 0.5)
    # High persistence + thoroughness -> good, but too much speed hurts
    base = 0.5 * persistence + 0.3 * thoroughness + 0.2 * (1.0 - abs(speed - 0.5))
    return min(1.0, max(0.0, base))


def efficiency_score(genome: AgentGenome) -> float:
    """Score based on speed and caution balance — fast but not reckless."""
    speed = genome.get("speed", 0.5)
    caution = genome.get("caution", 0.5)
    # Optimal is moderate-high speed with moderate caution
    speed_score = 1.0 - abs(speed - 0.7)
    caution_score = 1.0 - abs(caution - 0.5)
    return 0.6 * speed_score + 0.4 * caution_score


def accuracy_score(genome: AgentGenome) -> float:
    """Score based on thoroughness and caution — careful, detailed agents are accurate."""
    thoroughness = genome.get("thoroughness", 0.5)
    caution = genome.get("caution", 0.5)
    adaptability = genome.get("adaptability", 0.5)
    return 0.4 * thoroughness + 0.35 * caution + 0.25 * adaptability


def adaptability_score(genome: AgentGenome) -> float:
    """Score based on adaptability and creativity traits."""
    adaptability = genome.get("adaptability", 0.5)
    creativity = genome.get("creativity", 0.5)
    speed = genome.get("speed", 0.5)
    return 0.4 * adaptability + 0.35 * creativity + 0.25 * (1.0 - abs(speed - 0.6))


BUILTIN_METRICS: Dict[str, Callable[[AgentGenome], float]] = {
    FitnessMetric.TASK_COMPLETION: task_completion_score,
    FitnessMetric.EFFICIENCY: efficiency_score,
    FitnessMetric.ACCURACY: accuracy_score,
    FitnessMetric.ADAPTABILITY: adaptability_score,
}


@dataclass
class FitnessEvaluator:
    """Evaluates genomes against one or more fitness metrics."""
    metrics: Dict[str, Callable[[AgentGenome], float]] = field(
        default_factory=lambda: dict(BUILTIN_METRICS)
    )
    weights: Optional[Dict[str, float]] = None

    def evaluate(self, genome: AgentGenome) -> FitnessResult:
        """Evaluate a single genome and return a FitnessResult."""
        scores: Dict[str, float] = {}
        for name, func in self.metrics.items():
            scores[name] = round(func(genome), 6)

        if self.weights:
            total_w = sum(self.weights.get(k, 0.0) for k in scores)
            if total_w > 0:
                composite = sum(
                    scores[k] * self.weights.get(k, 0.0) for k in scores
                ) / total_w
            else:
                composite = sum(scores.values()) / len(scores)
        else:
            composite = sum(scores.values()) / len(scores) if scores else 0.0

        return FitnessResult(
            genome_id=genome.genome_id,
            scores=scores,
            composite=round(composite, 6),
        )

    def evaluate_population(
        self, genomes: Sequence[AgentGenome]
    ) -> List[FitnessResult]:
        """Evaluate all genomes in a sequence."""
        return [self.evaluate(g) for g in genomes]
