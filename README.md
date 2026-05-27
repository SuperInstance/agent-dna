# agent-dna

Genetic encoding for agent capabilities and behavior traits. Define agent genomes, evolve populations across generations, and track fitness and diversity — part of the [Cocapn fleet](https://github.com/Lucineer/the-fleet).

## Install

```bash
pip install -e .
```

Requires Python 3.10+. No external runtime dependencies — only `pytest` for the test suite.

## Quick Start

### Create and inspect a genome

```python
from agent_dna import AgentGenome

# Random genome with all built-in traits
genome = AgentGenome.random()
print(genome)
# AgentGenome(id='a3f1b2c4', gen=0, genes=8)

# Access trait values
print(genome.get("persistence"))   # 0.0 – 1.0
print(genome.get("creativity"))    # 0.0 – 1.0

# Create from explicit values
custom = AgentGenome.from_values({
    "persistence": 0.8,
    "creativity": 0.6,
    "caution": 0.7,
    "speed": 0.4,
    "verbosity": 0.3,
})
```

### Evolve a population

```python
from agent_dna import Population, evolve, SelectionStrategy
from agent_dna.fitness import FitnessEvaluator

# Create a random population of 30 agents
pop = Population.random(size=30)

# Run 50 generations of evolution
stats = pop.run(generations=50, elite_count=3)

# Check the best agent
best = pop.best()
print(f"Best genome: {best}")
print(f"Fitness improvement: {stats[0].best_fitness:.3f} → {stats[-1].best_fitness:.3f}")

# Print summary
print(pop.summary())
```

### Serialize and deserialize

```python
# To JSON
json_str = genome.to_json()

# From JSON
restored = AgentGenome.from_json(json_str)

# Dict roundtrip
data = genome.to_dict()
clone = AgentGenome.from_dict(data)
```

### Custom fitness metrics

```python
from agent_dna.fitness import FitnessEvaluator

evaluator = FitnessEvaluator(
    metrics={
        "speed_focus": lambda g: g.get("speed", 0.0),
        "caution_penalty": lambda g: 1.0 - g.get("caution", 0.0),
    },
    weights={"speed_focus": 0.7, "caution_penalty": 0.3},
)

result = evaluator.evaluate(genome)
print(result.scores)       # {"speed_focus": 0.4, "caution_penalty": 0.5}
print(result.composite)    # weighted average
```

### Mutation and crossover

```python
# Mutate (returns a new genome)
child = genome.mutate(rate=0.2, sigma=0.1)

# Crossover two parents
parent1 = AgentGenome.from_values({"speed": 0.2, "caution": 0.9})
parent2 = AgentGenome.from_values({"speed": 0.8, "caution": 0.1})
child1, child2 = parent1.crossover(parent2, method="uniform")
# or: parent1.crossover(parent2, method="blended", blend_alpha=0.5)
```

## Built-in Traits

| Trait | Category | Description |
|-------|----------|-------------|
| persistence | behavior | How long an agent continues before giving up |
| creativity | cognitive | Tendency to explore novel solutions |
| caution | behavior | Risk aversion before acting |
| speed | performance | Preference for fast execution |
| verbosity | social | Detail level in responses |
| adaptability | cognitive | How quickly it adjusts to changes |
| thoroughness | performance | Depth of analysis |
| cooperativeness | social | Willingness to collaborate |

All traits range from 0.0 to 1.0 with a default of 0.5.

## Architecture

```
agent_dna/
├── __init__.py       # Public API exports
├── traits.py         # Trait definitions and registry
├── genome.py         # AgentGenome with genes, mutation, crossover, serialization
├── fitness.py        # Fitness evaluation functions
├── evolution.py      # Selection strategies and evolve() loop
└── population.py     # Population management with diversity tracking
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## License

MIT

---

Part of the [Cocapn fleet](https://github.com/Lucineer/the-fleet). Built with [Cocapn](https://github.com/Lucineer/cocapn-ai).
