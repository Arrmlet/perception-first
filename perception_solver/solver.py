"""
ARC Solver v2 - Hybrid approach (Perception + Ensemble)

Features:
- Diff maps for clear feedback
- Temperature 1.0 for diversity
- Example shuffling
- Can run as single expert or parallel ensemble

Change MODEL in config.py to switch providers.
"""

import re
import json
import numpy as np
from typing import Optional

from config import MODEL, SOLVER_CONFIG
from llm import call_llm
from prompts import SOLVER_PROMPT, FEEDBACK_PROMPT, format_problem, format_problem_with_perception, format_feedback, format_diff
from sandbox import verify_code, execute_code
from logger import log_expert_iter


def parse_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def build_feedback_with_diff(result: dict, train_examples: list) -> tuple[str, float]:
    """Build feedback string with visual diff maps."""
    lines = []
    scores = []

    for i, r in enumerate(result.get("results", [])):
        if r["correct"]:
            lines.append(f"Example {i+1}: CORRECT")
            scores.append(1.0)
        elif r["error"]:
            lines.append(f"Example {i+1}: ERROR - {r['error'][:200]}")
            scores.append(0.0)
        else:
            expected = r["expected"]
            output = r["output"]

            if output is None:
                lines.append(f"Example {i+1}: No output produced")
                scores.append(0.0)
            else:
                exp_arr = np.array(expected)
                out_arr = np.array(output)

                if exp_arr.shape != out_arr.shape:
                    lines.append(f"Example {i+1}: WRONG SHAPE - expected {exp_arr.shape}, got {out_arr.shape}")
                    scores.append(0.0)
                else:
                    diff_count = np.sum(exp_arr != out_arr)
                    total = exp_arr.size
                    pct = 100 * diff_count / total
                    lines.append(f"Example {i+1}: WRONG - {diff_count}/{total} pixels ({pct:.1f}%)")

                    # Add visual diff map
                    lines.append(format_diff(expected, output))

                    scores.append(r.get("soft_score", 0.0))

    mean_score = float(np.mean(scores)) if scores else 0.0
    return "\n".join(lines), mean_score


def build_feedback(result: dict, train_examples: list) -> tuple[str, float]:
    """Build feedback string from verification result."""
    lines = []
    scores = []

    for i, r in enumerate(result.get("results", [])):
        if r["correct"]:
            lines.append(f"Example {i+1}: CORRECT")
            scores.append(1.0)
        elif r["error"]:
            lines.append(f"Example {i+1}: ERROR - {r['error'][:200]}")
            scores.append(0.0)
        else:
            # Wrong output
            expected = np.array(r["expected"])
            output = np.array(r["output"]) if r["output"] else None

            if output is None:
                lines.append(f"Example {i+1}: No output")
                scores.append(0.0)
            elif output.shape != expected.shape:
                lines.append(f"Example {i+1}: WRONG SHAPE - expected {expected.shape}, got {output.shape}")
                scores.append(0.0)
            else:
                diff_count = np.sum(output != expected)
                total = expected.size
                pct = 100 * diff_count / total
                lines.append(f"Example {i+1}: WRONG - {diff_count}/{total} pixels ({pct:.1f}%)")

                # Show diff
                lines.append("  Expected vs Got (format: expected/got for wrong pixels):")
                for row_idx in range(expected.shape[0]):
                    row_str = "  "
                    for col_idx in range(expected.shape[1]):
                        e = expected[row_idx, col_idx]
                        o = output[row_idx, col_idx]
                        if e == o:
                            row_str += f"{e} "
                        else:
                            row_str += f"{e}/{o} "
                    lines.append(row_str)

                scores.append(r.get("soft_score", 0.0))

    mean_score = float(np.mean(scores)) if scores else 0.0
    return "\n".join(lines), mean_score


def solve_task(
    train_examples: list,
    test_inputs: list,
    model: str = MODEL,
    config: dict = SOLVER_CONFIG,
    task_id: str = "",
    logger=None,
    verbose: bool = True,
) -> dict:
    """
    Solve a single ARC task.

    Returns: {
        "success": bool,
        "test_outputs": list,
        "iterations": int,
        "prompt_tokens": int,
        "completion_tokens": int,
        "best_score": float,
        "solutions": list
    }
    """
    max_iterations = config.get("max_iterations", 10)
    temperature = config.get("temperature", 1.0)
    max_tokens = config.get("max_tokens", 16384)
    shuffle_examples = config.get("shuffle_examples", True)
    max_previous = config.get("max_previous_solutions", 5)
    timeout_s = config.get("timeout_s", 5.0)

    solutions = []  # List of {code, feedback, score}
    best_score = 0.0
    best_code = None
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for iteration in range(max_iterations):
        if verbose and logger:
            logger.subheader(f"Iteration {iteration + 1}/{max_iterations}")

        # Build prompt
        problem_str = format_problem(
            train_examples,
            test_inputs,
            shuffle=shuffle_examples,
            seed=iteration
        )
        prompt = SOLVER_PROMPT.replace("$$problem$$", problem_str)

        # Add feedback from previous attempts
        if solutions:
            feedback_str = format_feedback(solutions, max_previous)
            prompt += "\n\n" + FEEDBACK_PROMPT.replace("$$feedback$$", feedback_str)

        messages = [{"role": "user", "content": prompt}]

        # Call LLM
        try:
            response, prompt_tokens, completion_tokens = call_llm(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        except Exception as e:
            if verbose and logger:
                logger.error(f"LLM error: {e}")
            continue

        if verbose and logger:
            logger.info(f"Tokens: +{prompt_tokens + completion_tokens:,} (total: {total_prompt_tokens + total_completion_tokens:,})")

        # Parse code
        code = parse_code(response)
        if not code:
            if verbose and logger:
                logger.warning("No code found in response")
            continue

        if verbose and logger:
            logger.debug(f"Code extracted ({len(code)} chars)")

        # Verify on training examples
        result = verify_code(code, train_examples, timeout_s)
        nc = result["num_correct"]
        nt = result["num_total"]

        # Build feedback
        feedback, score = build_feedback(result, train_examples)
        solutions.append({"code": code, "feedback": feedback, "score": score})

        # Log iteration result
        if verbose and logger:
            if nc == nt:
                result_status = "correct"
            elif nc > 0:
                result_status = "partial"
            else:
                result_status = "wrong"
            logger.iteration(iteration + 1, max_iterations, result_status,
                           prompt_tokens + completion_tokens, score)

        # Track best
        if score > best_score:
            best_score = score
            best_code = code
            if verbose and logger:
                logger.success(f"NEW BEST: {score:.2f}")

        # Check if solved
        if result["all_correct"]:
            if verbose and logger:
                logger.success("ALL TRAINING EXAMPLES CORRECT!")

            # Execute on test inputs
            test_outputs = []
            for test_inp in test_inputs:
                exec_result = execute_code(code, test_inp, timeout_s)
                if exec_result["success"]:
                    test_outputs.append(exec_result["output"])
                else:
                    test_outputs.append(None)

            return {
                "success": True,
                "test_outputs": test_outputs,
                "iterations": iteration + 1,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "best_score": best_score,
                "best_code": code,
                "solutions": solutions,
            }

    # Failed to solve
    if verbose and logger:
        logger.warning(f"No solution found after {max_iterations} iterations. Best score: {best_score:.2f}")

    # Try best code on test anyway
    test_outputs = []
    if best_code:
        for test_inp in test_inputs:
            exec_result = execute_code(best_code, test_inp, timeout_s)
            if exec_result["success"]:
                test_outputs.append(exec_result["output"])
            else:
                test_outputs.append(None)

    return {
        "success": False,
        "test_outputs": test_outputs,
        "iterations": max_iterations,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "best_score": best_score,
        "best_code": best_code,
        "solutions": solutions,
    }


def solve_task_single(
    train_examples: list,
    test_inputs: list,
    task_id: str = "",
    seed: int = 0,
    model: str = MODEL,
    config: dict = SOLVER_CONFIG,
    use_perception: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Single expert solver with specific seed for parallel execution.

    This is a lightweight version without logging for use in parallel_solver.
    The seed is used to:
    1. Shuffle examples differently each expert
    2. Create diversity across experts

    Args:
        use_perception: If True, include perception v10 analysis (STATE → AFFORDANCE)
    """
    max_iterations = config.get("max_iterations", 10)
    temperature = config.get("temperature", 1.0)
    max_tokens = config.get("max_tokens", 16384)
    shuffle_examples = config.get("shuffle_examples", True)
    max_previous = config.get("max_previous_solutions", 5)
    timeout_s = config.get("timeout_s", 5.0)

    solutions = []
    best_score = 0.0
    best_code = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    ever_got_partial = False  # Track if we ever got at least 1 correct on train

    for iteration in range(max_iterations):
        # Early termination: if 5 rounds with 0 correct, give up
        if iteration >= 5 and not ever_got_partial:
            if verbose:
                from logger import cprint, Colors
                cprint(f"    Expert {seed}: terminated (0 correct after 5 rounds)", Colors.DIM)
            break
        # Use seed + iteration for unique shuffling per expert per iteration
        iter_seed = seed * 1000 + iteration

        # Build prompt with or without perception
        if use_perception:
            perception_str, grids_str = format_problem_with_perception(
                train_examples,
                test_inputs,
                shuffle=shuffle_examples,
                seed=iter_seed
            )
            prompt = SOLVER_PROMPT.replace("$$perception$$", perception_str).replace("$$problem$$", grids_str)
        else:
            problem_str = format_problem(
                train_examples,
                test_inputs,
                shuffle=shuffle_examples,
                seed=iter_seed
            )
            prompt = SOLVER_PROMPT.replace("$$perception$$", "(Perception disabled)").replace("$$problem$$", problem_str)

        # Add feedback from previous attempts
        if solutions:
            feedback_str = format_feedback(solutions, max_previous)
            prompt += "\n\n" + FEEDBACK_PROMPT.replace("$$feedback$$", feedback_str)

        messages = [{"role": "user", "content": prompt}]

        # Always save logs with model/reasoning info
        import os
        from datetime import datetime

        # Build log directory: logs/{model}_{reasoning}/
        provider, model_name = model.split("/", 1)
        reasoning_level = config.get("reasoning_effort", "none") if provider == "openai" else f"think{config.get('thinking_budget', 0)}"
        log_dir = os.path.join(os.path.dirname(__file__), "logs", f"{model_name}_{reasoning_level}")
        os.makedirs(log_dir, exist_ok=True)

        # Save prompt
        prompt_file = os.path.join(log_dir, f"{task_id}_expert{seed}_iter{iteration}_prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(f"# Model: {model}\n# Reasoning: {reasoning_level}\n# Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(prompt)

        # Call LLM
        try:
            response, prompt_tokens, completion_tokens = call_llm(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=verbose,
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            # Save response
            response_file = os.path.join(log_dir, f"{task_id}_expert{seed}_iter{iteration}_response.txt")
            with open(response_file, "w") as f:
                f.write(f"# Model: {model}\n# Reasoning: {reasoning_level}\n# Tokens: prompt={prompt_tokens}, completion={completion_tokens}\n\n")
                f.write(response)
        except Exception as e:
            if verbose:
                log_expert_iter(seed, iteration + 1, max_iterations, "error")
            print(f"  Expert {seed} iter {iteration + 1}: LLM error: {e}")
            continue

        # Parse code
        code = parse_code(response)
        if not code:
            if verbose:
                log_expert_iter(seed, iteration + 1, max_iterations, "error")
            continue

        # Verify on training examples
        result = verify_code(code, train_examples, timeout_s)
        nc = result["num_correct"]
        nt = result["num_total"]

        # Build feedback with diff maps
        feedback, score = build_feedback_with_diff(result, train_examples)
        solutions.append({"code": code, "feedback": feedback, "score": score})

        # Track if we ever got at least 1 correct
        if nc > 0:
            ever_got_partial = True

        # Verbose logging
        if verbose:
            if nc == nt:
                status = "success"
            elif nc > 0:
                status = "partial"
            else:
                status = "error"
            log_expert_iter(seed, iteration + 1, max_iterations, status)

        # Track best
        if score > best_score:
            best_score = score
            best_code = code

        # Check if solved
        if result["all_correct"]:
            # Execute on test inputs
            test_outputs = []
            for test_inp in test_inputs:
                exec_result = execute_code(code, test_inp, timeout_s)
                if exec_result["success"]:
                    test_outputs.append(exec_result["output"])
                else:
                    test_outputs.append(None)

            return {
                "success": True,
                "test_outputs": test_outputs,
                "iterations": iteration + 1,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "best_score": best_score,
                "best_code": code,
            }

    # Failed to solve - try best code on test anyway
    test_outputs = []
    if best_code:
        for test_inp in test_inputs:
            exec_result = execute_code(best_code, test_inp, timeout_s)
            if exec_result["success"]:
                test_outputs.append(exec_result["output"])
            else:
                test_outputs.append(None)

    return {
        "success": False,
        "test_outputs": test_outputs,
        "iterations": max_iterations,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "best_score": best_score,
        "best_code": best_code,
    }
