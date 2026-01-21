"""
Unified LLM interface - supports Anthropic, OpenAI, and Gemini with one API.
"""

import os
from typing import Optional
from config import API_KEYS, SOLVER_CONFIG


def get_client(model: str):
    """Get the appropriate client for the model."""
    provider = model.split("/")[0]

    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=API_KEYS["anthropic"] or None)

    elif provider == "openai":
        os.environ["OPENAI_API_KEY"] = API_KEYS["openai"]
        from openai import OpenAI
        return OpenAI(timeout=600.0)  # 10 min timeout for xhigh reasoning

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=API_KEYS["gemini"])
        return genai

    elif provider == "xai":
        from openai import OpenAI
        return OpenAI(
            api_key=API_KEYS["xai"],
            base_url="https://api.x.ai/v1"
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm(
    model: str,
    messages: list,
    temperature: float = 1.0,
    max_tokens: int = 16384,
    verbose: bool = False,
) -> tuple[str, int, int]:
    """
    Call LLM with unified interface.

    Returns: (response_text, prompt_tokens, completion_tokens)
    """
    provider, model_name = model.split("/", 1)
    client = get_client(model)

    if provider == "anthropic":
        return _call_anthropic(client, model_name, messages, temperature, max_tokens, verbose)
    elif provider == "openai":
        return _call_openai(client, model_name, messages, temperature, max_tokens, verbose)
    elif provider == "gemini":
        return _call_gemini(client, model_name, messages, temperature, max_tokens, verbose)
    elif provider == "xai":
        return _call_xai(client, model_name, messages, temperature, max_tokens, verbose)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(client, model: str, messages: list, temperature: float, max_tokens: int, verbose: bool = False):
    """Call Anthropic Claude API."""
    from logger import cprint, Colors

    # Separate system message
    system = ""
    user_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            user_messages.append(msg)

    # Check if thinking should be enabled
    thinking_budget = SOLVER_CONFIG.get("thinking_budget", 0)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": user_messages,
    }

    if system:
        kwargs["system"] = system

    if thinking_budget > 0:
        # Extended thinking for Claude models (Opus, Sonnet)
        kwargs["temperature"] = 1.0  # Required for thinking
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }
    else:
        kwargs["temperature"] = temperature

    response = client.messages.create(**kwargs)

    # Extract text and thinking from response
    text = ""
    thinking = ""
    for block in response.content:
        if hasattr(block, 'thinking'):
            thinking = block.thinking
        elif hasattr(block, 'text'):
            text += block.text

    # Show thinking in verbose mode
    if verbose and thinking:
        cprint("\n💭 THINKING:", Colors.MAGENTA + Colors.BOLD)
        cprint("-" * 50, Colors.DIM)
        # Truncate if too long
        if len(thinking) > 2000:
            cprint(thinking[:2000] + "\n... [truncated]", Colors.DIM)
        else:
            cprint(thinking, Colors.DIM)
        cprint("-" * 50, Colors.DIM)

    prompt_tokens = response.usage.input_tokens
    completion_tokens = response.usage.output_tokens

    # Return thinking appended to text for logging (separated by marker)
    if thinking:
        text = f"<thinking>\n{thinking}\n</thinking>\n\n{text}"

    return text, prompt_tokens, completion_tokens


def _call_openai(client, model: str, messages: list, temperature: float, max_tokens: int, verbose: bool = False):
    """Call OpenAI GPT API."""
    import os
    from logger import cprint, Colors

    reasoning_effort = SOLVER_CONFIG.get("reasoning_effort", "medium")

    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }

    # GPT-5 series uses reasoning_effort instead of temperature
    if "gpt-5" in model.lower():
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["temperature"] = 1.0  # Required for reasoning models
    else:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)

    text = response.choices[0].message.content or ""

    # Extract reasoning content if available (GPT-5 reasoning models)
    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)

    # Debug: log full response structure
    if os.environ.get("DEBUG_LLM"):
        cprint(f"\n📊 RESPONSE DEBUG:", Colors.CYAN + Colors.BOLD)
        cprint(f"  message attrs: {dir(response.choices[0].message)}", Colors.DIM)
        if hasattr(response, 'usage'):
            usage = response.usage
            cprint(f"  usage: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}", Colors.DIM)
            if hasattr(usage, 'completion_tokens_details'):
                cprint(f"  completion_details: {usage.completion_tokens_details}", Colors.DIM)
        cprint(f"  reasoning_content: {reasoning[:500] if reasoning else 'None'}", Colors.DIM)

    # Show reasoning in verbose mode or DEBUG_LLM
    if (verbose or os.environ.get("DEBUG_LLM")) and reasoning:
        cprint("\n💭 REASONING:", Colors.MAGENTA + Colors.BOLD)
        cprint("-" * 50, Colors.DIM)
        # Truncate if too long
        if len(reasoning) > 2000:
            cprint(reasoning[:2000] + "\n... [truncated]", Colors.DIM)
        else:
            cprint(reasoning, Colors.DIM)
        cprint("-" * 50, Colors.DIM)

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    return text, prompt_tokens, completion_tokens


def _call_gemini(client, model: str, messages: list, temperature: float, max_tokens: int, verbose: bool = False):
    """Call Google Gemini API."""
    # Convert messages to Gemini format
    gemini_model = client.GenerativeModel(model)

    # Build prompt from messages
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"{content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"

    response = gemini_model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
    )

    text = response.text
    # Gemini doesn't expose token counts easily
    prompt_tokens = 0
    completion_tokens = 0

    return text, prompt_tokens, completion_tokens


def _call_xai(client, model: str, messages: list, temperature: float, max_tokens: int, verbose: bool = False):
    """Call XAI Grok API (OpenAI-compatible)."""
    from logger import cprint, Colors

    reasoning_effort = SOLVER_CONFIG.get("reasoning_effort", "high")

    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }

    # Grok-4 reasoning models
    if "reasoning" in model.lower():
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**kwargs)

    text = response.choices[0].message.content or ""

    # Extract reasoning content if available
    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)

    # Show reasoning in verbose mode
    if verbose and reasoning:
        cprint("\n💭 REASONING:", Colors.MAGENTA + Colors.BOLD)
        cprint("-" * 50, Colors.DIM)
        if len(reasoning) > 2000:
            cprint(reasoning[:2000] + "\n... [truncated]", Colors.DIM)
        else:
            cprint(reasoning, Colors.DIM)
        cprint("-" * 50, Colors.DIM)

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    return text, prompt_tokens, completion_tokens
