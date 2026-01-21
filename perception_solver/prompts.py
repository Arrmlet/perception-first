"""
Prompts for Perception-First ARC Solver

Key features:
- Slim perception (~150 tokens) for visual structure
- Kitchen metaphor: STATE → AFFORDANCE
- Few-shot examples
- Diff maps for feedback
- Parallel experts with voting
"""

from perception_slim import perceive_slim


SOLVER_PROMPT = '''You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles.

## Approach (Think Like a Chef)

When you see a grid, think like a chef seeing ingredients:
1. **SCAN**: What objects/patterns exist? What STATE is each in?
2. **AFFORDANCES**: What ACTIONS are possible? (extract, rotate, tile, fill, repair, split)
3. **PLAN**: What sequence of steps transforms input → output?
4. **IMPLEMENT**: Write clean code

## Output Format

1. Brief explanation (1-2 sentences)
2. Python code in ```python block with `def transform(grid: np.ndarray) -> np.ndarray`

## Worked Examples

**Example 1: Border Replacement**

Input:
```
[[1, 1, 1],
 [1, 0, 1],
 [1, 1, 1]]
```

Output:
```
[[0, 0, 0],
 [0, 1, 0],
 [0, 0, 0]]
```

Explanation: Swap the border and center values.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    result = grid.copy()
    result[grid == 1] = 0
    result[grid == 0] = 1
    return result
```

**Example 2: Rotation**

Input:
```
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```

Output:
```
[[9, 8, 7],
 [6, 5, 4],
 [3, 2, 1]]
```

Explanation: Rotate the grid 180 degrees.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 2)
```

**Example 3: Extraction**

Input:
```
[[0, 0, 0, 0, 0],
 [0, 1, 2, 3, 0],
 [0, 4, 5, 6, 0],
 [0, 0, 0, 0, 0]]
```

Output:
```
[[1, 2, 3],
 [4, 5, 6]]
```

Explanation: Extract the non-zero region.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    return grid[rows][:, cols]
```

---

## PERCEPTION ANALYSIS

$$perception$$

---

## YOUR PROBLEM

$$problem$$
'''

FEEDBACK_PROMPT = '''
## Previous Attempts

Your previous solutions didn't fully work. Learn from these attempts:

$$feedback$$

## Instructions
- Study what went WRONG (look at the diff maps)
- Check the AFFORDANCES - are you using the right action for each object's state?
- Try a DIFFERENT approach if the current one keeps failing
- Remember: multi-step solutions often work better than single jumps

Provide an improved solution.
'''


def format_problem_with_perception(train_examples: list, test_inputs: list, shuffle: bool = False, seed: int = 0) -> tuple:
    """Format the problem with perception analysis.

    Returns (perception_str, grids_str) to be inserted into prompt.
    """
    import numpy as np

    examples = list(train_examples)

    if shuffle and len(examples) > 1:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(examples))
        examples = [examples[i] for i in perm]

    # Generate perception analysis
    perception_str = perceive_slim(examples, test_inputs)

    # Format raw grids (compact, since perception already shows annotated versions)
    lines = []
    for i, ex in enumerate(examples, 1):
        inp = ex['input']
        out = ex['output']

        lines.append(f"### Training Example {i}")
        lines.append(f"Input ({len(inp)}x{len(inp[0])}):")
        lines.append("```")
        for row in inp:
            lines.append("".join(str(c) for c in row))
        lines.append("```")
        lines.append(f"Output ({len(out)}x{len(out[0])}):")
        lines.append("```")
        for row in out:
            lines.append("".join(str(c) for c in row))
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")

    for i, inp in enumerate(test_inputs, 1):
        lines.append(f"### Test Input {i} ({len(inp)}x{len(inp[0])})")
        lines.append("```")
        for row in inp:
            lines.append("".join(str(c) for c in row))
        lines.append("```")
        lines.append("")

    grids_str = "\n".join(lines)

    return perception_str, grids_str


def format_problem(train_examples: list, test_inputs: list, shuffle: bool = False, seed: int = 0) -> str:
    """Format the problem for the prompt (legacy, without perception)."""
    import numpy as np

    examples = list(train_examples)

    if shuffle and len(examples) > 1:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(examples))
        examples = [examples[i] for i in perm]

    lines = []

    for i, ex in enumerate(examples, 1):
        inp = ex['input']
        out = ex['output']

        lines.append(f"### Training Example {i}")
        lines.append("")
        lines.append("Input:")
        lines.append("```")
        for row in inp:
            lines.append(" ".join(str(c) for c in row))
        lines.append("```")
        lines.append("")
        lines.append("Output:")
        lines.append("```")
        for row in out:
            lines.append(" ".join(str(c) for c in row))
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")

    for i, inp in enumerate(test_inputs, 1):
        lines.append(f"### Test Input {i}")
        lines.append("")
        lines.append("```")
        for row in inp:
            lines.append(" ".join(str(c) for c in row))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def format_diff(expected: list, actual: list) -> str:
    """Create a visual diff map showing differences."""
    import numpy as np

    exp = np.array(expected)
    act = np.array(actual)

    if exp.shape != act.shape:
        return f"Shape mismatch: expected {exp.shape}, got {act.shape}"

    lines = []
    lines.append("Diff (. = correct, X = wrong):")
    lines.append("```")

    for r in range(exp.shape[0]):
        row_str = ""
        for c in range(exp.shape[1]):
            if exp[r, c] == act[r, c]:
                row_str += ". "
            else:
                row_str += "X "
        lines.append(f"{r:2}| {row_str}")

    lines.append("```")

    # Count errors
    wrong = np.sum(exp != act)
    total = exp.size
    lines.append(f"Wrong pixels: {wrong}/{total} ({100*wrong/total:.1f}%)")

    return "\n".join(lines)


def format_feedback(solutions: list, max_solutions: int = 5) -> str:
    """Format previous solutions as feedback with diff maps."""
    if not solutions:
        return ""

    # Sort by score descending (best first)
    sorted_solutions = sorted(solutions, key=lambda x: x['score'], reverse=True)[:max_solutions]

    lines = []
    for i, sol in enumerate(sorted_solutions, 1):
        score_pct = sol['score'] * 100
        lines.append(f"### Attempt {i} (score: {score_pct:.0f}%)")
        lines.append("")
        lines.append("```python")
        lines.append(sol['code'])
        lines.append("```")
        lines.append("")
        lines.append("**Result:**")
        lines.append(sol['feedback'])
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
