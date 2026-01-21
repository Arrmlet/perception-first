# Perception-First ARC Solver

A novel approach to solving ARC-AGI-2 puzzles by focusing on **perception design** rather than reasoning complexity.

## Key Insight

> LLMs can reason, but they can't see. The bottleneck isn't reasoning capability—it's how we present visual information.

From baseline **26.7%** to **~50%** accuracy on ARC-AGI-2 by improving perception, not model size.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  PERCEPTION-FIRST SOLVER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │  PERCEPTION     │  ← Slim (~150 tokens)                     │
│  │  (perception_   │  ← Task type detection                    │
│  │   slim.py)      │  ← Diff maps                              │
│  │                 │  ← Key transitions                         │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PARALLEL EXPERTS (4-8)                      │   │
│  │                                                          │   │
│  │   Temperature 1.0 for diversity                          │   │
│  │   Each expert iterates up to 10 times                    │   │
│  │   Diff-based feedback for learning                       │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  VOTING         │  ← Consensus from passing solutions       │
│  │                 │  ← Early stopping on correct answer       │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Results

| Setup | ARC-AGI-2 Accuracy |
|-------|-------------------|
| GPT-5.2 medium baseline | 26.7% |
| GPT-5.2 + perception + 4 experts | **~50%** |

## Installation

```bash
cd perception_solver
pip install anthropic openai google-generativeai numpy scipy
```

Set API keys:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # optional
export GEMINI_API_KEY="your-key"     # optional
```

## Usage

```bash
cd perception_solver

# Run single task
python run_parallel.py --task 0934a4d8 --experts 4

# Run first N tasks
python run_parallel.py --count 10 --experts 4

# Run all evaluation tasks
python run_parallel.py --experts 4
```

## Configuration

Edit `config.py` to change model and settings:

```python
MODEL = "openai/gpt-5.2"  # or anthropic/claude-opus-4-5

SOLVER_CONFIG = {
    "max_iterations": 10,
    "temperature": 1.0,
    "reasoning_effort": "medium",  # for GPT-5
    "thinking_budget": 0,          # for Claude (0 = disabled)
}
```

## Research

See [PERCEPTION_RESEARCH.md](PERCEPTION_RESEARCH.md) for the theoretical foundation based on:
- Biological vision (V1→V2→V4→IT hierarchy)
- Gibson's affordance theory
- Gestalt principles
- Predictive coding

## Key Files

| File | Purpose |
|------|---------|
| `run_parallel.py` | Main entry point |
| `perception_slim.py` | Slim perception (~150 tokens) |
| `parallel_solver.py` | Multi-expert coordination |
| `solver.py` | Single expert solver |
| `llm.py` | Unified LLM interface (OpenAI, Anthropic, Gemini) |
| `prompts.py` | Prompt templates |
| `sandbox.py` | Safe code execution |

## Citation

```bibtex
@article{perception-first-arc-2025,
  title={Perception-First Approach to ARC-AGI-2:
         LLMs Can Reason, But They Can't See},
  year={2025}
}
```

## License

MIT
