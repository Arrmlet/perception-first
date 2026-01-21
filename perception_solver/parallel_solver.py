"""
Parallel Solver with Voting - Perception-First approach:
- Slim perception (~150 tokens) for structure detection
- Multiple parallel experts for diversity
- Voting mechanism for consensus

Usage:
    python parallel_solver.py --task 0934a4d8 --experts 4
    python parallel_solver.py --count 10 --experts 8
"""

import asyncio
import json
import hashlib
from typing import Optional
from collections import defaultdict

from config import MODEL, SOLVER_CONFIG
from solver import solve_task_single
from sandbox import execute_code
from logger import log_header, log_expert_start, log_expert_done, log_voting, log_task_result


async def solve_task_async(
    train_examples: list,
    test_inputs: list,
    task_id: str,
    seed: int,
    model: str,
    config: dict,
    use_perception: bool = True,
    verbose: bool = False,
) -> dict:
    """Async wrapper for single expert solve."""
    if verbose:
        log_expert_start(seed, config.get('_num_experts', 8))

    # Run in thread pool since solve_task_single is sync
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: solve_task_single(
            train_examples=train_examples,
            test_inputs=test_inputs,
            task_id=task_id,
            seed=seed,
            model=model,
            config=config,
            use_perception=use_perception,
            verbose=verbose,
        )
    )
    result['expert_seed'] = seed

    if verbose:
        tokens = result.get('prompt_tokens', 0) + result.get('completion_tokens', 0)
        log_expert_done(seed, result.get('success', False), result.get('iterations', 0), result.get('best_score', 0), tokens)

    return result


async def solve_with_experts(
    train_examples: list,
    test_inputs: list,
    task_id: str = "",
    model: str = MODEL,
    config: dict = SOLVER_CONFIG,
    num_experts: int = 8,
    use_perception: bool = True,
    verbose: bool = True,
    test_expected: list = None,  # For early stopping when correct answer found
) -> dict:
    """
    Run multiple experts in parallel and vote on results.

    Returns: {
        "success": bool,
        "test_outputs": list (top voted),
        "attempt_2": list (second best),
        "iterations": int (total across experts),
        "prompt_tokens": int,
        "completion_tokens": int,
        "expert_results": list,
        "vote_counts": dict,
    }
    """
    if verbose:
        log_header(f"Task: {task_id} | Model: {model} | Experts: {num_experts}")

    # Pass num_experts to config for logging
    config_with_meta = {**config, '_num_experts': num_experts}

    # Launch all experts as asyncio tasks
    async_tasks = [
        asyncio.create_task(solve_task_async(
            train_examples=train_examples,
            test_inputs=test_inputs,
            task_id=task_id,
            seed=i,
            model=model,
            config=config_with_meta,
            use_perception=use_perception,
            verbose=verbose,
        ))
        for i in range(num_experts)
    ]

    if verbose:
        print(f"\n🚀 Launching {num_experts} experts in parallel...\n")

    # Check results as they complete, stop early if correct test output found
    expert_results = []
    early_stop = False
    correct_result = None

    for coro in asyncio.as_completed(async_tasks):
        result = await coro
        expert_results.append(result)

        # Check if this result is correct on test (if test_expected provided)
        if test_expected and result.get('success') and result.get('test_outputs'):
            import numpy as np
            from logger import cprint, Colors
            is_correct = True
            for pred, exp in zip(result['test_outputs'], test_expected):
                if pred is None:
                    is_correct = False
                    break
                try:
                    pred_arr = np.array(pred)
                    exp_arr = np.array(exp)
                    if pred_arr.shape != exp_arr.shape or not np.array_equal(pred_arr, exp_arr):
                        is_correct = False
                        break
                except:
                    is_correct = False
                    break

            # Log test result (always show, important info)
            if is_correct:
                cprint(f"    → Expert {result.get('expert_seed')} test: ✓ CORRECT", Colors.GREEN)
            else:
                cprint(f"    → Expert {result.get('expert_seed')} test: ✗ wrong", Colors.DIM)

            if is_correct:
                early_stop = True
                correct_result = result
                from logger import cprint, Colors
                tokens_used = result.get('prompt_tokens', 0) + result.get('completion_tokens', 0)
                cprint(f"\n🎯 EARLY STOP: Expert {result.get('expert_seed')} found CORRECT answer!", Colors.GREEN + Colors.BOLD)
                cprint(f"   ✓ Solved in {result.get('iterations', 0)} iterations | {tokens_used:,} tokens", Colors.GREEN)
                cprint(f"   ✓ Cancelling {sum(1 for t in async_tasks if not t.done())} remaining experts...", Colors.GREEN)
                # Cancel remaining tasks
                for t in async_tasks:
                    if not t.done():
                        t.cancel()
                break

    # Wait for cancelled tasks to finish (they'll raise CancelledError)
    for t in async_tasks:
        if not t.done():
            try:
                await t
            except asyncio.CancelledError:
                pass

    # Calculate totals
    total_iterations = sum(r.get('iterations', 0) for r in expert_results)
    total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in expert_results)
    total_completion_tokens = sum(r.get('completion_tokens', 0) for r in expert_results)

    # If early stopped with correct answer, use that directly
    if early_stop and correct_result:
        return {
            "success": True,
            "test_outputs": correct_result['test_outputs'],
            "attempt_2": None,
            "iterations": total_iterations,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "best_score": 1.0,
            "best_code": correct_result.get('best_code'),
            "expert_results": expert_results,
            "vote_counts": {"early_stop": 1},
            "early_stop": True,
        }

    # Aggregate results using voting
    aggregated = aggregate_with_voting(expert_results, verbose=verbose)

    return {
        "success": aggregated['success'],
        "test_outputs": aggregated['test_outputs'],
        "attempt_2": aggregated.get('attempt_2'),
        "iterations": total_iterations,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "best_score": aggregated['best_score'],
        "best_code": aggregated.get('best_code'),
        "expert_results": expert_results,
        "vote_counts": aggregated['vote_counts'],
        "early_stop": False,
    }


def hash_output(output: list) -> str:
    """Create hash of output for grouping."""
    if output is None:
        return "none"
    try:
        return hashlib.md5(json.dumps(output).encode()).hexdigest()[:16]
    except:
        return "error"


def aggregate_with_voting(expert_results: list, verbose: bool = True) -> dict:
    """
    Aggregate expert results using voting mechanism.

    Strategy:
    1. Group results by test output (hash)
    2. Separate passing (all training correct) from failing
    3. Rank by vote count
    4. Diversity-first: take one from each group
    5. Return top 2 as attempt_1 and attempt_2
    """
    # Separate passing and failing results
    passing = []  # All training examples correct
    failing = []  # Some training examples wrong

    for result in expert_results:
        if result.get('success'):
            passing.append(result)
        else:
            failing.append(result)

    if verbose:
        log_voting(len(passing), len(failing), 0)  # Will update unique count below

    # Group by test output
    passing_buckets = defaultdict(list)
    failing_buckets = defaultdict(list)

    for result in passing:
        outputs = result.get('test_outputs', [])
        key = tuple(hash_output(o) for o in outputs)
        passing_buckets[key].append(result)

    for result in failing:
        outputs = result.get('test_outputs', [])
        key = tuple(hash_output(o) for o in outputs)
        failing_buckets[key].append(result)

    if verbose:
        print(f"  📦 Unique outputs: {len(passing_buckets)} passing, {len(failing_buckets)} failing")

    # Rank passing buckets by vote count
    passing_ranked = sorted(
        passing_buckets.items(),
        key=lambda x: (-len(x[1]), -max(r.get('best_score', 0) for r in x[1]))
    )

    # Rank failing buckets by vote count, then best score
    failing_ranked = sorted(
        failing_buckets.items(),
        key=lambda x: (-len(x[1]), -max(r.get('best_score', 0) for r in x[1]))
    )

    # Build final order (diversity-first)
    final_order = []
    vote_counts = {}

    # First: one from each passing bucket
    for key, bucket in passing_ranked:
        best = max(bucket, key=lambda r: r.get('best_score', 0))
        final_order.append(best)
        vote_counts[key] = len(bucket)
        if verbose:
            print(f"  ✓ Candidate (votes={len(bucket)}): score={best.get('best_score', 0):.2f}")

    # Then: one from each failing bucket (in case passing fails on test)
    for key, bucket in failing_ranked:
        best = max(bucket, key=lambda r: r.get('best_score', 0))
        final_order.append(best)
        vote_counts[key] = len(bucket)

    # Get top 2 results
    if final_order:
        best_result = final_order[0]
        attempt_2 = final_order[1]['test_outputs'] if len(final_order) > 1 else None

        return {
            "success": best_result.get('success', False),
            "test_outputs": best_result.get('test_outputs', []),
            "attempt_2": attempt_2,
            "best_score": best_result.get('best_score', 0),
            "best_code": best_result.get('best_code'),
            "vote_counts": dict(vote_counts),
        }
    else:
        return {
            "success": False,
            "test_outputs": [],
            "attempt_2": None,
            "best_score": 0,
            "best_code": None,
            "vote_counts": {},
        }


# Sync wrapper for compatibility
def solve_task(
    train_examples: list,
    test_inputs: list,
    task_id: str = "",
    model: str = MODEL,
    config: dict = SOLVER_CONFIG,
    num_experts: int = 8,
    verbose: bool = True,
) -> dict:
    """Sync wrapper for solve_with_experts."""
    return asyncio.run(
        solve_with_experts(
            train_examples=train_examples,
            test_inputs=test_inputs,
            task_id=task_id,
            model=model,
            config=config,
            num_experts=num_experts,
            verbose=verbose,
        )
    )
