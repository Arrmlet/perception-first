"""
Logging system for ARC Solver v2.

Features:
- Colored terminal output
- Per-task log files
- Summary log file
- JSON results
- Real-time expert/iteration progress
"""

import os
import sys
import json
import datetime
from typing import Optional
from enum import Enum


class Colors:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    ERROR = 4


# =============================================================================
# Simple console logging for verbose mode (no file I/O)
# =============================================================================

_USE_COLORS = sys.stdout.isatty()


def cprint(msg: str, color: str = "", end: str = "\n"):
    """Print with color if terminal supports it."""
    if _USE_COLORS and color:
        print(f"{color}{msg}{Colors.RESET}", end=end, flush=True)
    else:
        print(msg, end=end, flush=True)


def log_header(text: str, char: str = "=", width: int = 70):
    """Print a header."""
    line = char * width
    cprint(line, Colors.CYAN)
    cprint(text, Colors.BOLD + Colors.CYAN)
    cprint(line, Colors.CYAN)


def log_subheader(text: str, char: str = "-", width: int = 50):
    """Print a subheader."""
    cprint(f"{char * 3} {text} {char * 3}", Colors.BLUE)


def log_expert_start(expert_id: int, num_experts: int):
    """Log expert starting."""
    cprint(f"  🤖 Expert {expert_id}/{num_experts} started", Colors.CYAN)


def log_expert_iter(expert_id: int, iteration: int, max_iter: int, status: str = "running"):
    """Log expert iteration progress."""
    if status == "success":
        symbol = "✓"
        color = Colors.GREEN
    elif status == "partial":
        symbol = "◐"
        color = Colors.YELLOW
    elif status == "error":
        symbol = "✗"
        color = Colors.RED
    else:
        symbol = "→"
        color = Colors.DIM

    cprint(f"    Expert {expert_id} [{iteration}/{max_iter}] {symbol}", color)


def log_expert_done(expert_id: int, success: bool, iterations: int, score: float, tokens: int):
    """Log expert completion."""
    if success:
        cprint(f"  ✓ Expert {expert_id}: SOLVED in {iterations} iter | score={score:.2f} | tokens={tokens:,}", Colors.GREEN)
    else:
        cprint(f"  ✗ Expert {expert_id}: failed | best={score:.2f} | tokens={tokens:,}", Colors.RED)


def log_voting(passing: int, failing: int, unique_outputs: int):
    """Log voting aggregation."""
    cprint(f"\n📊 Voting: {passing} passing, {failing} failing, {unique_outputs} unique outputs", Colors.MAGENTA)


def log_task_result(task_id: str, correct: bool, tokens: int, best_score: float):
    """Log final task result."""
    if correct:
        cprint(f"\n🎉 CORRECT: {task_id} | tokens={tokens:,} | score={best_score:.2f}", Colors.BG_GREEN + Colors.BOLD)
    else:
        cprint(f"\n❌ FAILED: {task_id} | tokens={tokens:,} | score={best_score:.2f}", Colors.RED + Colors.BOLD)


class Logger:
    """Logger with colored output and file logging."""

    def __init__(self, output_dir: str, task_id: Optional[str] = None):
        self.output_dir = output_dir
        self.task_id = task_id
        self.log_file = None
        self.use_colors = sys.stdout.isatty()

        os.makedirs(output_dir, exist_ok=True)

        if task_id:
            tasks_dir = os.path.join(output_dir, "tasks")
            os.makedirs(tasks_dir, exist_ok=True)
            log_path = os.path.join(tasks_dir, f"{task_id}.log")
        else:
            log_path = os.path.join(output_dir, "summary.log")

        self.log_file = open(log_path, 'w', encoding='utf-8')

    def _color(self, text: str, color: str) -> str:
        """Apply color if terminal supports it."""
        if self.use_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _write(self, message: str, color: str = "", to_file: bool = True):
        """Write to terminal and optionally to file."""
        # Terminal (with color)
        if color and self.use_colors:
            print(f"{color}{message}{Colors.RESET}")
        else:
            print(message)

        # File (without color)
        if to_file and self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def header(self, text: str, char: str = "=", width: int = 70):
        """Print a header."""
        line = char * width
        self._write(line, Colors.CYAN)
        self._write(text, Colors.BOLD + Colors.CYAN)
        self._write(line, Colors.CYAN)

    def subheader(self, text: str, char: str = "-", width: int = 70):
        """Print a subheader."""
        line = char * width
        self._write(line, Colors.BLUE)
        self._write(text, Colors.BLUE)
        self._write(line, Colors.BLUE)

    def info(self, message: str):
        """Info message."""
        self._write(message)

    def success(self, message: str):
        """Success message (green)."""
        self._write(f"✓ {message}", Colors.GREEN)

    def warning(self, message: str):
        """Warning message (yellow)."""
        self._write(f"⚠ {message}", Colors.YELLOW)

    def error(self, message: str):
        """Error message (red)."""
        self._write(f"✗ {message}", Colors.RED)

    def debug(self, message: str):
        """Debug message (dim)."""
        self._write(f"  {message}", Colors.DIM)

    def code(self, code: str, language: str = "python"):
        """Print code block."""
        self._write(f"```{language}")
        self._write(code)
        self._write("```")

    def grid(self, grid: list, label: str = ""):
        """Print a grid."""
        if label:
            self._write(f"{label} ({len(grid)}x{len(grid[0])}):", Colors.CYAN)
        for row in grid:
            self._write(''.join(str(c) for c in row), Colors.DIM)

    def diff(self, expected: list, actual: list):
        """Print diff between two grids."""
        import numpy as np
        exp_arr = np.array(expected)
        act_arr = np.array(actual)

        if exp_arr.shape != act_arr.shape:
            self.error(f"Shape mismatch: expected {exp_arr.shape}, got {act_arr.shape}")
            return

        self._write("Diff (expected/actual for wrong pixels):", Colors.YELLOW)
        for r in range(exp_arr.shape[0]):
            row_str = ""
            for c in range(exp_arr.shape[1]):
                e, a = exp_arr[r, c], act_arr[r, c]
                if e == a:
                    row_str += f"{e} "
                else:
                    row_str += self._color(f"{e}/{a} ", Colors.RED) if self.use_colors else f"{e}/{a} "
            print(row_str)  # Direct print for color mixing
            if self.log_file:
                # File version without colors
                file_row = ""
                for c in range(exp_arr.shape[1]):
                    e, a = exp_arr[r, c], act_arr[r, c]
                    if e == a:
                        file_row += f"{e} "
                    else:
                        file_row += f"{e}/{a} "
                self.log_file.write(file_row + "\n")

    def iteration(self, num: int, total: int, result: str, tokens: int, score: float):
        """Print iteration result."""
        if result == "correct":
            color = Colors.GREEN
            symbol = "✓"
        elif result == "partial":
            color = Colors.YELLOW
            symbol = "◐"
        else:
            color = Colors.RED
            symbol = "✗"

        msg = f"  [{num}/{total}] {symbol} tokens={tokens:,} score={score:.2f}"
        self._write(msg, color)

    def task_result(self, task_id: str, status: str, iterations: int, tokens: int, best_score: float):
        """Print task result line."""
        if status == "CORRECT":
            color = Colors.GREEN + Colors.BOLD
        elif status == "VERIFIED":
            color = Colors.YELLOW
        else:
            color = Colors.RED

        msg = f"{status:8} {task_id} | iter={iterations} | tokens={tokens:,} | best={best_score:.2f}"
        self._write(msg, color)

    def close(self):
        """Close log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


class RunLogger:
    """Manages logging for a batch run."""

    def __init__(self, model: str, config: dict):
        self.model = model
        self.config = config
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        model_short = model.replace("/", "_").replace("-", "")[:20]
        self.output_dir = os.path.join(
            os.path.dirname(__file__),
            "output",
            f"{model_short}_{self.timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.results = []
        self.main_logger = Logger(self.output_dir)
        self.current_task_logger = None

    def start_task(self, task_id: str) -> Logger:
        """Start logging for a new task."""
        if self.current_task_logger:
            self.current_task_logger.close()

        self.current_task_logger = Logger(self.output_dir, task_id)
        return self.current_task_logger

    def end_task(self, result: dict):
        """End logging for current task."""
        if self.current_task_logger:
            self.current_task_logger.close()
            self.current_task_logger = None

        self.results.append(result)
        self._save_results()

    def _save_results(self):
        """Save results to JSON."""
        results_path = os.path.join(self.output_dir, "results.json")

        correct = sum(1 for r in self.results if r.get('correct'))
        verified = sum(1 for r in self.results if r.get('success'))
        total_tokens = sum(r.get('prompt_tokens', 0) + r.get('completion_tokens', 0) for r in self.results)

        data = {
            'model': self.model,
            'config': self.config,
            'timestamp': self.timestamp,
            'total': len(self.results),
            'correct': correct,
            'verified': verified,
            'tokens': total_tokens,
            'output_dir': self.output_dir,
            'results': self.results,
        }

        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Print final summary."""
        n = max(len(self.results), 1)
        correct = sum(1 for r in self.results if r.get('correct'))
        verified = sum(1 for r in self.results if r.get('success'))
        total_tokens = sum(r.get('prompt_tokens', 0) + r.get('completion_tokens', 0) for r in self.results)

        self.main_logger.header("SUMMARY")
        self.main_logger.info(f"Model: {self.model}")
        self.main_logger.info(f"Tasks: {len(self.results)}")
        self.main_logger.info(f"Verified: {verified}/{len(self.results)} ({100*verified/n:.1f}%)")
        self.main_logger.info(f"Correct: {correct}/{len(self.results)} ({100*correct/n:.1f}%)")
        self.main_logger.info(f"Tokens: {total_tokens:,} ({total_tokens//n:,}/task avg)")
        self.main_logger.info(f"Output: {self.output_dir}")

        self.main_logger.subheader("Results")
        for r in self.results:
            status = "CORRECT" if r.get('correct') else ("VERIFIED" if r.get('success') else "FAILED")
            tokens = r.get('prompt_tokens', 0) + r.get('completion_tokens', 0)
            self.main_logger.task_result(
                r.get('task_id', '?'),
                status,
                r.get('iterations', 0),
                tokens,
                r.get('best_score', 0)
            )

    def close(self):
        """Close all loggers."""
        if self.current_task_logger:
            self.current_task_logger.close()
        self.main_logger.close()
