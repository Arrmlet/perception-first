"""
Sandbox for executing transform code safely.
"""

import json
import subprocess
import sys
import tempfile
import os


def run_code(code: str, input_grid: list, timeout_s: float = 5.0) -> tuple[bool, str, list | None]:
    """
    Run transform code in a subprocess sandbox.

    Returns: (success, error_or_output_str, output_grid_or_none)
    """
    script = f'''
import json
import numpy as np
from scipy import ndimage
import sys

{code}

if __name__ == "__main__":
    data = json.load(sys.stdin)
    grid = np.array(data["input"])
    try:
        result = transform(grid)
        if isinstance(result, np.ndarray):
            result = result.tolist()
        print(json.dumps({{"ok": True, "result": result}}))
    except Exception as e:
        print(json.dumps({{"ok": False, "error": str(e)}}))
'''

    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "transform.py")
        with open(script_path, "w") as f:
            f.write(script)

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                input=json.dumps({"input": input_grid}),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=td,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return False, "Timeout", None

        if proc.returncode != 0:
            error = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"
            return False, error, None

        try:
            result = json.loads(proc.stdout)
            if result.get("ok"):
                return True, "", result.get("result")
            else:
                return False, result.get("error", "Unknown error"), None
        except json.JSONDecodeError:
            return False, f"Invalid JSON output: {proc.stdout[:200]}", None


def verify_code(code: str, train_examples: list, timeout_s: float = 5.0) -> dict:
    """
    Verify code against all training examples.

    Returns: {
        "all_correct": bool,
        "num_correct": int,
        "num_total": int,
        "results": [{
            "correct": bool,
            "error": str | None,
            "expected": list,
            "output": list | None,
            "soft_score": float
        }]
    }
    """
    import numpy as np

    results = []
    num_correct = 0

    for ex in train_examples:
        inp = ex['input']
        expected = ex['output']

        success, error, output = run_code(code, inp, timeout_s)

        if not success:
            results.append({
                "correct": False,
                "error": error,
                "expected": expected,
                "output": None,
                "soft_score": 0.0
            })
            continue

        # Compare output to expected
        try:
            out_arr = np.array(output)
            exp_arr = np.array(expected)

            if out_arr.shape == exp_arr.shape and np.array_equal(out_arr, exp_arr):
                correct = True
                soft_score = 1.0
                num_correct += 1
            else:
                correct = False
                if out_arr.shape == exp_arr.shape:
                    soft_score = float(np.mean(out_arr == exp_arr))
                else:
                    soft_score = 0.0

            results.append({
                "correct": correct,
                "error": None,
                "expected": expected,
                "output": output,
                "soft_score": soft_score
            })
        except Exception as e:
            results.append({
                "correct": False,
                "error": str(e),
                "expected": expected,
                "output": output,
                "soft_score": 0.0
            })

    return {
        "all_correct": num_correct == len(train_examples),
        "num_correct": num_correct,
        "num_total": len(train_examples),
        "results": results
    }


def execute_code(code: str, test_input: list, timeout_s: float = 5.0) -> dict:
    """
    Execute code on test input.

    Returns: {
        "success": bool,
        "error": str | None,
        "output": list | None
    }
    """
    success, error, output = run_code(code, test_input, timeout_s)

    return {
        "success": success,
        "error": error if not success else None,
        "output": output
    }
