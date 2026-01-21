"""
Configuration for ARC Solver v2

Change MODEL to switch between providers/models.
"""

# ============================================================
# CHANGE THIS LINE TO SWITCH MODELS
# ============================================================

MODEL = "anthropic/claude-opus-4-5"

# Other options:
# MODEL = "anthropic/claude-opus-4-5-20250514"
# MODEL = "anthropic/claude-haiku-4-5-20250514"
# MODEL = "openai/gpt-5.2"
# MODEL = "openai/gpt-5.1"
# MODEL = "openai/gpt-5"
# MODEL = "gemini/gemini-2.5-pro"
# MODEL = "gemini/gemini-3-pro-preview"
# MODEL = "xai/grok-4"
# MODEL = "xai/grok-4-fast"

# ============================================================
# SOLVER SETTINGS
# ============================================================

SOLVER_CONFIG = {
    # Iteration settings
    "max_iterations": 10,
    "temperature": 1.0,  # High for diversity across experts
    "shuffle_examples": True,  # Randomize example order each iteration

    # Token limits
    "max_tokens": 16384,

    # Feedback settings
    "max_previous_solutions": 5,  # Show top N previous attempts
    "selection_probability": 1.0,  # Probability of including previous solutions

    # Execution
    "timeout_s": 5.0,  # Code execution timeout

    # Thinking/Reasoning (model-specific)
    "thinking_budget": 4000,  # For Claude extended thinking (0 = disabled)
    "reasoning_effort": "medium",  # For OpenAI GPT-5 (none/low/medium/high/xhigh)
}

# ============================================================
# API KEYS (set via environment or here)
# ============================================================

import os

API_KEYS = {
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "gemini": os.environ.get("GEMINI_API_KEY", ""),
    "xai": os.environ.get("XAI_API_KEY", ""),
}
