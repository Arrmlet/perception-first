"""
Perception Slim - Key insights only (~500 tokens instead of 4000)

Based on research findings:
1. Kitchen Metaphor: STATE → AFFORDANCE (what actions are legal)
2. Diff maps: Visual pattern discovery
3. Before/After rows: See exact changes
4. Task family: Guide hypothesis direction

What we DON'T include:
- SOURCE FOUND (only helps 3.3% of tasks)
- 400+ object detection (noise)
- Verbose MARKER labels everywhere
"""

import numpy as np
from typing import List, Dict
from collections import Counter


def perceive_slim(train_examples: List[Dict], test_inputs: List[List[List[int]]]) -> str:
    """Generate slim perception - only the key insights."""
    lines = []

    lines.append("## PERCEPTION")
    lines.append("")

    # Analyze task type
    task_type = classify_task(train_examples)
    lines.append(f"**Task Type:** {task_type}")
    lines.append("")

    # Process each example
    for idx, ex in enumerate(train_examples):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])

        lines.append(f"### Example {idx + 1}")
        lines.append(f"Size: {inp.shape[0]}x{inp.shape[1]} → {out.shape[0]}x{out.shape[1]}")

        # Key objects (max 3, only if meaningful)
        key_objects = find_key_objects(inp, out)
        if key_objects:
            lines.append(f"Key objects: {key_objects}")

        # For same-size: show diff map and before/after
        if inp.shape == out.shape:
            diff_info = analyze_diff(inp, out)
            if diff_info:
                lines.append(diff_info)

        lines.append("")

    # Cross-example pattern (one line)
    pattern = find_cross_pattern(train_examples)
    if pattern:
        lines.append(f"**Pattern:** {pattern}")
        lines.append("")

    # Affordance hint (one line)
    affordance = suggest_affordance(train_examples, task_type)
    if affordance:
        lines.append(f"**Try:** {affordance}")

    return "\n".join(lines)


def classify_task(train_examples: List[Dict]) -> str:
    """Classify task type in one word."""
    sizes = []
    for ex in train_examples:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        sizes.append((inp.shape, out.shape))

    all_same = all(s[0] == s[1] for s in sizes)
    all_smaller = all(s[1][0] <= s[0][0] and s[1][1] <= s[0][1] for s in sizes)
    all_larger = all(s[1][0] >= s[0][0] and s[1][1] >= s[0][1] for s in sizes)

    if all_same:
        # Check if it's repair (periodic pattern)
        ex = train_examples[0]
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        diff_pct = np.sum(inp != out) / inp.size
        if diff_pct < 0.3:
            return "EDIT (sparse changes)"
        else:
            return "TRANSFORM (same size)"
    elif all_smaller and not all_same:
        return "EXTRACTION (output smaller)"
    elif all_larger and not all_same:
        return "EXPANSION (output larger)"
    else:
        return "MIXED"


def find_key_objects(inp: np.ndarray, out: np.ndarray) -> str:
    """Find max 3 key objects worth mentioning."""
    bg = int(np.bincount(inp.ravel()).argmax())

    objects = []
    for color in np.unique(inp):
        if color == bg:
            continue

        mask = (inp == color)
        count = np.sum(mask)
        total = inp.size
        pct = count / total

        # Only mention if significant
        if pct > 0.05:  # More than 5% of grid
            rows, cols = np.where(mask)
            h = rows.max() - rows.min() + 1
            w = cols.max() - cols.min() + 1

            # Check if it's a solid rectangle
            is_solid = count == h * w

            if is_solid and h > 1 and w > 1:
                objects.append(f"color-{int(color)} rect ({h}x{w})")
            elif h == 1 or w == 1:
                objects.append(f"color-{int(color)} line")

    # Check for dividers
    h, w = inp.shape
    for c in range(w):
        col = inp[:, c]
        if len(np.unique(col)) == 1 and col[0] != bg:
            objects.append(f"vertical divider at col {c}")
            break

    for r in range(h):
        row = inp[r, :]
        if len(np.unique(row)) == 1 and row[0] != bg:
            objects.append(f"horizontal divider at row {r}")
            break

    return ", ".join(objects[:3]) if objects else ""


def analyze_diff(inp: np.ndarray, out: np.ndarray) -> str:
    """Analyze diff for same-size grids - compact output."""
    diff_mask = inp != out
    diff_count = np.sum(diff_mask)

    if diff_count == 0:
        return "No changes (identical)"

    total = inp.size
    pct = 100 * diff_count / total

    lines = []
    lines.append(f"Changes: {diff_count}/{total} ({pct:.0f}%)")

    # Compact diff map (only if small enough)
    h, w = inp.shape
    if h <= 15 and w <= 15:
        lines.append("Diff: (. = same, X = changed)")
        for r in range(h):
            row_str = ""
            for c in range(w):
                row_str += "X" if diff_mask[r, c] else "."
            lines.append(f"  {row_str}")

    # Before/After for changed rows (max 3)
    changed_rows = np.where(np.any(diff_mask, axis=1))[0]
    if len(changed_rows) > 0 and len(changed_rows) <= 5:
        lines.append("Before/After:")
        for r in changed_rows[:3]:
            in_row = ''.join(str(int(x)) for x in inp[r])
            out_row = ''.join(str(int(x)) for x in out[r])
            lines.append(f"  Row {r}: {in_row} → {out_row}")

    # Color transitions
    transitions = Counter()
    rows, cols = np.where(diff_mask)
    for r, c in zip(rows, cols):
        transitions[(int(inp[r, c]), int(out[r, c]))] += 1

    if transitions:
        top = transitions.most_common(2)
        trans_str = ", ".join(f"{a}→{b} ({n}x)" for (a, b), n in top)
        lines.append(f"Transitions: {trans_str}")

    return "\n".join(lines)


def find_cross_pattern(train_examples: List[Dict]) -> str:
    """Find one-line pattern across examples."""
    patterns = []

    # Check for consistent color removal
    removed_colors = None
    for ex in train_examples:
        inp_colors = set(np.unique(ex['input']))
        out_colors = set(np.unique(ex['output']))
        removed = inp_colors - out_colors
        if removed_colors is None:
            removed_colors = removed
        else:
            removed_colors &= removed

    if removed_colors:
        patterns.append(f"color {list(removed_colors)} always removed")

    # Check for size ratio consistency
    ratios = []
    for ex in train_examples:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        if out.shape[0] > 0 and out.shape[1] > 0:
            ratios.append((inp.shape[0] / out.shape[0], inp.shape[1] / out.shape[1]))

    if len(set(ratios)) == 1 and ratios[0] != (1.0, 1.0):
        patterns.append(f"consistent ratio {ratios[0][0]:.1f}x{ratios[0][1]:.1f}")

    # Check for periodic pattern in output
    for ex in train_examples:
        out = np.array(ex['output'])
        period = detect_period(out)
        if period:
            patterns.append(f"output has period {period}")
            break

    return "; ".join(patterns) if patterns else ""


def detect_period(grid: np.ndarray) -> str:
    """Detect if grid has periodic pattern."""
    h, w = grid.shape

    # Check row periodicity
    for period in range(2, min(w // 2 + 1, 6)):
        if w % period != 0:
            continue
        is_periodic = True
        for r in range(h):
            template = grid[r, :period]
            for i in range(period, w, period):
                if not np.array_equal(grid[r, i:i+period], template):
                    is_periodic = False
                    break
            if not is_periodic:
                break
        if is_periodic:
            return f"cols period={period}"

    # Check column periodicity
    for period in range(2, min(h // 2 + 1, 6)):
        if h % period != 0:
            continue
        is_periodic = True
        for c in range(w):
            template = grid[:period, c]
            for i in range(period, h, period):
                if not np.array_equal(grid[i:i+period, c], template):
                    is_periodic = False
                    break
            if not is_periodic:
                break
        if is_periodic:
            return f"rows period={period}"

    return ""


def suggest_affordance(train_examples: List[Dict], task_type: str) -> str:
    """Suggest one action based on task type."""
    if "EXTRACTION" in task_type:
        return "Find the region to extract (look for unique color/shape)"
    elif "EXPANSION" in task_type:
        return "Tile or repeat the input pattern"
    elif "EDIT" in task_type:
        return "Find what triggers the change (neighbor, position, color)"
    elif "TRANSFORM" in task_type:
        return "Look for rotation, flip, or color mapping"
    else:
        return ""


# Test
if __name__ == "__main__":
    import json

    # Load a real task
    with open('../ARC-AGI-2/data/evaluation/0934a4d8.json') as f:
        task = json.load(f)

    train = task['train']
    test = [t['input'] for t in task['test']]

    result = perceive_slim(train, test)
    print(f"Output length: {len(result)} chars (~{len(result)//4} tokens)")
    print()
    print(result)
