#!/usr/bin/env python3
"""
Perception-First ARC Solver - Parallel Ensemble Runner

This is the main entry point for the perception-first approach:
- Slim perception (~150 tokens) for visual structure
- Multiple parallel experts for diversity
- Diff maps for feedback
- Voting mechanism for consensus

Usage:
    python run_parallel.py                              # Run all eval tasks with 4 experts
    python run_parallel.py --count 10                   # Run first 10 tasks
    python run_parallel.py --count 10 --experts 8       # Run with 8 experts
    python run_parallel.py --task 0934a4d8              # Run specific task
    python run_parallel.py --task 0934a4d8 --experts 2  # Quick test with 2 experts
"""

import os
import sys
import json
import glob
import argparse
import asyncio
import datetime
import numpy as np
from tqdm import tqdm

from config import MODEL, SOLVER_CONFIG
from parallel_solver import solve_with_experts


def load_task(task_path: str) -> dict:
    """Load a task from JSON file."""
    with open(task_path, 'r') as f:
        return json.load(f)


def score_output(predicted: list, expected: list) -> float:
    """Score a single output. Returns 1.0 if correct, 0.0 otherwise."""
    if predicted is None:
        return 0.0
    try:
        pred_arr = np.array(predicted)
        exp_arr = np.array(expected)
        return 1.0 if pred_arr.shape == exp_arr.shape and np.array_equal(pred_arr, exp_arr) else 0.0
    except:
        return 0.0


async def run_single_task(task_path: str, num_experts: int = 4, use_perception: bool = True, verbose: bool = False) -> dict:
    """Run parallel solver on a single task."""
    task_id = os.path.basename(task_path).replace('.json', '')
    task = load_task(task_path)

    train = task['train']
    test = task['test']
    test_inputs = [t['input'] for t in test]
    test_expected = [t['output'] for t in test if 'output' in t]

    result = await solve_with_experts(
        train_examples=train,
        test_inputs=test_inputs,
        task_id=task_id,
        model=MODEL,
        config=SOLVER_CONFIG,
        num_experts=num_experts,
        use_perception=use_perception,
        verbose=verbose,
        test_expected=test_expected if test_expected else None,
    )

    # Score test outputs (both attempts)
    correct = False
    for i, t in enumerate(test):
        if 'output' in t:
            expected = t['output']

            # Check attempt 1
            if i < len(result["test_outputs"]):
                pred1 = result["test_outputs"][i]
                if score_output(pred1, expected) == 1.0:
                    correct = True
                    print(f"  Test {i+1}: CORRECT (attempt 1)")
                    continue

            # Check attempt 2
            if result.get("attempt_2") and i < len(result["attempt_2"]):
                pred2 = result["attempt_2"][i]
                if score_output(pred2, expected) == 1.0:
                    correct = True
                    print(f"  Test {i+1}: CORRECT (attempt 2)")
                    continue

            print(f"  Test {i+1}: WRONG (both attempts)")

    result["task_id"] = task_id
    result["correct"] = correct

    # Summary
    print(f"\n{'='*60}")
    if result['correct']:
        print(f"RESULT: CORRECT")
    else:
        print(f"RESULT: FAILED")
    print(f"Experts: {num_experts}")
    print(f"Total iterations: {result['iterations']}")
    print(f"Total tokens: {result['prompt_tokens'] + result['completion_tokens']:,}")
    print(f"Best score: {result['best_score']:.2f}")
    print(f"Vote counts: {result.get('vote_counts', {})}")
    print(f"{'='*60}")

    return result


async def run_batch(data_dir: str, count: int = None, num_experts: int = 4, use_perception: bool = True, resume_dir: str = None, verbose: bool = False):
    """Run parallel solver on multiple tasks with checkpoint/resume support."""
    task_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if count:
        task_files = task_files[:count]

    # Resume from existing run or create new output directory
    if resume_dir and os.path.exists(resume_dir):
        output_dir = resume_dir
        # Load existing results
        results_path = os.path.join(output_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                existing_data = json.load(f)
                results = existing_data.get("results", [])
                completed_tasks = {r["task_id"] for r in results}
                print(f"Resuming from {output_dir}")
                print(f"Already completed: {len(completed_tasks)} tasks")
        else:
            results = []
            completed_tasks = set()
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = MODEL.replace("/", "_").replace("-", "")[:20]
        output_dir = os.path.join(
            os.path.dirname(__file__),
            "output",
            f"{model_short}_exp{num_experts}_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)
        results = []
        completed_tasks = set()

    # Calculate initial stats from resumed results
    correct_count = sum(1 for r in results if r.get("correct"))
    total_tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in results)

    print("="*60)
    print(f"ARC Solver v2 - Parallel Ensemble")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Experts: {num_experts}")
    print(f"Perception: {'ENABLED (slim)' if use_perception else 'DISABLED (raw grids)'}")
    print(f"Tasks: {len(task_files)} ({len(completed_tasks)} already done)")
    print(f"Max iterations: {SOLVER_CONFIG.get('max_iterations', 10)} per expert")
    print(f"Temperature: {SOLVER_CONFIG.get('temperature', 1.0)}")
    print(f"Output: {output_dir}")
    print("="*60)

    for i, task_path in enumerate(task_files):
        task_id = os.path.basename(task_path).replace('.json', '')

        # Skip already completed tasks
        if task_id in completed_tasks:
            print(f"\n[{i+1}/{len(task_files)}] Task: {task_id} - SKIPPED (already done)")
            continue

        print(f"\n[{i+1}/{len(task_files)}] Task: {task_id}")
        print("-"*40)

        task = load_task(task_path)
        train = task['train']
        test = task['test']
        test_inputs = [t['input'] for t in test]
        test_expected = [t['output'] for t in test if 'output' in t]

        result = await solve_with_experts(
            train_examples=train,
            test_inputs=test_inputs,
            task_id=task_id,
            model=MODEL,
            config=SOLVER_CONFIG,
            num_experts=num_experts,
            use_perception=use_perception,
            verbose=verbose,
            test_expected=test_expected if test_expected else None,
        )

        # Score test outputs (check both attempts)
        task_correct = True
        for j, t in enumerate(test):
            if 'output' in t:
                expected = t['output']
                attempt1_correct = False
                attempt2_correct = False

                if j < len(result["test_outputs"]):
                    pred1 = result["test_outputs"][j]
                    attempt1_correct = score_output(pred1, expected) == 1.0

                if result.get("attempt_2") and j < len(result["attempt_2"]):
                    pred2 = result["attempt_2"][j]
                    attempt2_correct = score_output(pred2, expected) == 1.0

                if not (attempt1_correct or attempt2_correct):
                    task_correct = False

        result["task_id"] = task_id
        result["correct"] = task_correct
        results.append(result)

        if task_correct:
            correct_count += 1
            status = "CORRECT"
        elif result["success"]:
            status = "VERIFIED"
        else:
            status = "FAILED"

        tokens = result["prompt_tokens"] + result["completion_tokens"]
        total_tokens += tokens

        # Print task result
        passing_experts = len([r for r in result.get('expert_results', []) if r.get('success')])
        print(f"  {status} | experts_pass={passing_experts}/{num_experts} | "
              f"tokens={tokens:,} | best={result['best_score']:.2f}")

        # Save progress
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump({
                "model": MODEL,
                "config": SOLVER_CONFIG,
                "num_experts": num_experts,
                "total": len(results),
                "correct": correct_count,
                "tokens": total_tokens,
                "results": [
                    {k: v for k, v in r.items() if k not in ('expert_results', 'vote_counts')}
                    for r in results
                ],
            }, f, indent=2)

    # Final summary
    n = len(task_files)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Experts: {num_experts}")
    print(f"Tasks: {n}")
    print(f"Correct: {correct_count}/{n} ({100*correct_count/n:.1f}%)")
    print(f"Tokens: {total_tokens:,} ({total_tokens//n:,}/task avg)")
    print(f"Cost estimate: ~${total_tokens * 0.000003:.2f} (at $3/1M tokens)")
    print(f"Output: {output_dir}")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(description="ARC Solver v2 - Parallel Ensemble")
    parser.add_argument("--task", type=str, help="Run specific task (ID or path)")
    parser.add_argument("--count", type=int, help="Number of tasks to run")
    parser.add_argument("--experts", type=int, default=8, help="Number of parallel experts (default: 8)")
    parser.add_argument("--no-perception", action="store_true", help="Disable perception (use raw grids only)")
    parser.add_argument("--resume", type=str, help="Resume from existing output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable detailed colorful logging")
    parser.add_argument("--data-dir", type=str,
                       default=os.path.join(os.path.dirname(__file__), "..", "ARC-AGI-2", "data", "evaluation"),
                       help="Directory containing task JSON files")

    args = parser.parse_args()
    use_perception = not args.no_perception
    resume_dir = args.resume
    verbose = args.verbose

    if args.task:
        # Single task
        if os.path.exists(args.task):
            task_path = args.task
        else:
            task_path = os.path.join(args.data_dir, f"{args.task}.json")
            if not os.path.exists(task_path):
                print(f"Task not found: {args.task}")
                sys.exit(1)

        asyncio.run(run_single_task(task_path, num_experts=args.experts, use_perception=use_perception, verbose=verbose))
    else:
        # Batch run
        asyncio.run(run_batch(args.data_dir, count=args.count, num_experts=args.experts, use_perception=use_perception, resume_dir=resume_dir, verbose=verbose))


if __name__ == "__main__":
    main()
