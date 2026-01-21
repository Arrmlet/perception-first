# Reasoning Models: A Comprehensive Guide

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Standard LLMs vs Reasoning Models](#standard-llms-vs-reasoning-models)
3. [Training Techniques Deep Dive](#training-techniques-deep-dive)
4. [Chain-of-Thought: Prompting vs Native](#chain-of-thought-prompting-vs-native)
5. [Provider Implementations](#provider-implementations)
6. [Inference-Time Compute Scaling](#inference-time-compute-scaling)
7. [When to Use What](#when-to-use-what)
8. [Cost-Performance Tradeoffs](#cost-performance-tradeoffs)

---

## The Big Picture

### The Problem with Standard LLMs

Standard LLMs generate text token-by-token, predicting "what comes next" based on patterns learned during training. This works great for:
- Text completion
- Summarization
- Translation
- Simple Q&A

But it struggles with:
- Multi-step reasoning
- Math problems
- Logic puzzles
- Planning
- Tasks requiring "thinking before answering"

**Why?** Because the model commits to an answer immediately. If the first token it generates is wrong, it's stuck on a bad path.

### The Insight: Let Models "Think"

Humans don't solve complex problems instantly. We:
1. Break problems into steps
2. Try approaches, backtrack if wrong
3. Verify our work
4. Revise our thinking

**Reasoning models** give LLMs this same capability by:
- Allowing extended "thinking" before answering
- Training them to generate reasoning traces
- Rewarding correct reasoning, not just correct answers

---

## Standard LLMs vs Reasoning Models

### Standard LLM (e.g., GPT-4, Claude 3.5)

```
Input: "What is 17 * 24?"
Output: "408"
```

The model immediately outputs an answer. It might be right, might be wrong. No visible reasoning.

**Training objective**: Predict next token based on training data (web text, books, etc.)

### Reasoning Model (e.g., o1, Claude with thinking, DeepSeek R1)

```
Input: "What is 17 * 24?"

<thinking>
Let me break this down:
17 * 24 = 17 * (20 + 4)
       = 17 * 20 + 17 * 4
       = 340 + 68
       = 408

Let me verify: 408 / 17 = 24 ✓
</thinking>

Output: "408"
```

The model explicitly reasons through the problem before answering.

**Training objective**: Generate correct reasoning traces that lead to correct answers.

### Key Differences

| Aspect | Standard LLM | Reasoning Model |
|--------|--------------|-----------------|
| Output | Answer only | Thinking + Answer |
| Compute | Fixed per token | Variable (more thinking = more compute) |
| Training | Next-token prediction | Reinforcement learning on reasoning |
| Errors | Silent failures | Visible reasoning errors (debuggable) |
| Cost | Lower | Higher (more output tokens) |
| Latency | Fast | Slower (thinking takes time) |

---

## Training Techniques Deep Dive

### 1. Pre-training (Same for Both)

Both standard and reasoning models start the same way:

```
Objective: P(next_token | previous_tokens)

Training data: Trillions of tokens from internet, books, code
Method: Self-supervised learning
Scale: Thousands of GPUs, months of training
```

This gives the model general knowledge and language ability.

### 2. Supervised Fine-Tuning (SFT)

After pre-training, models are fine-tuned on curated examples.

**Standard LLM SFT:**
```
Human: What's the capital of France?
Assistant: The capital of France is Paris.
```

**Reasoning Model SFT:**
```
Human: What's the capital of France?
Assistant: <thinking>
The user is asking about France's capital city.
France is a country in Western Europe.
Its capital and largest city is Paris.
</thinking>
The capital of France is Paris.
```

For reasoning models, SFT data includes **reasoning traces** - examples of step-by-step thinking.

### 3. Reinforcement Learning from Human Feedback (RLHF)

Standard approach to align models with human preferences:

```
1. Generate multiple responses to same prompt
2. Humans rank responses (best to worst)
3. Train a "reward model" to predict human preferences
4. Use RL (PPO) to optimize LLM to maximize reward
```

**Reward Model** learns: "What makes a response good?"
- Helpful
- Harmless
- Honest

### 4. Process Reward Models (PRM) - Key for Reasoning!

Standard RLHF uses **Outcome Reward Models (ORM)**:
- Only judges the final answer
- Correct answer = high reward, wrong answer = low reward

**Problem**: Model can get right answer with wrong reasoning (lucky guess), or wrong answer despite good reasoning (arithmetic error).

**Process Reward Models** judge each reasoning step:

```
Step 1: "17 * 24 = 17 * (20 + 4)"     ✓ Correct decomposition
Step 2: "= 17 * 20 + 17 * 4"          ✓ Correct distribution
Step 3: "= 340 + 68"                   ✓ Correct multiplication
Step 4: "= 408"                        ✓ Correct addition
```

Each step gets a reward. Model learns to generate correct reasoning, not just correct answers.

**OpenAI's "Let's Verify Step by Step" paper (2023)** showed PRMs significantly outperform ORMs for math reasoning.

### 5. Reinforcement Learning Techniques

#### PPO (Proximal Policy Optimization)
- Standard RL algorithm used in RLHF
- Updates model weights to maximize reward
- Has "clipping" to prevent too-large updates (stability)

#### DPO (Direct Preference Optimization)
- Newer, simpler alternative to PPO
- Directly optimizes on preference data without separate reward model
- Used by many open-source reasoning models

#### GRPO (Group Relative Policy Optimization)
- Used by DeepSeek R1
- Compares groups of responses, not just pairs
- More sample-efficient

### 6. Self-Play and Synthetic Data

Modern reasoning models use **self-improvement**:

```
1. Model generates many reasoning traces for a problem
2. Verify which traces lead to correct answers
3. Use correct traces as training data
4. Repeat
```

This creates a flywheel:
- Better reasoning → More correct traces → Better training data → Better reasoning

**DeepSeek R1** used this extensively:
- Generated millions of reasoning traces
- Filtered for correctness
- Fine-tuned on the best ones

### 7. Distillation

Large reasoning models are expensive. **Distillation** transfers reasoning ability to smaller models:

```
Teacher (large model): Generates reasoning traces
Student (small model): Learns to mimic teacher's reasoning

Result: Smaller model gains reasoning ability without full RL training
```

DeepSeek released distilled versions: R1-7B, R1-14B, R1-32B from the full R1-671B.

---

## Chain-of-Thought: Prompting vs Native

### CoT Prompting (The Old Way)

Before reasoning models, we used **prompting tricks**:

```
Prompt: "Let's think step by step. What is 17 * 24?"

Response: "Let's break this down:
17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408"
```

**How it works:**
- The phrase "let's think step by step" biases the model toward verbose reasoning
- Model wasn't trained for this, just prompted
- Works sometimes, fails often

**Limitations:**
- Inconsistent (sometimes ignores the prompt)
- Reasoning quality varies wildly
- Model might "fake" reasoning (write steps that don't match its actual computation)
- No training signal for good reasoning

### Native Reasoning (The New Way)

Reasoning models are **trained** to think:

```
Input: "What is 17 * 24?"

<thinking>
[Model genuinely works through the problem]
[This is real computation, not performance]
</thinking>

Output: "408"
```

**Key differences:**

| CoT Prompting | Native Reasoning |
|---------------|------------------|
| Prompted behavior | Trained behavior |
| Visible to user | Often hidden (internal) |
| Unreliable | Consistent |
| Can be faked | Genuine reasoning |
| No compute scaling | More thinking = better results |

### The "Thinking" Block

Different providers handle thinking differently:

**OpenAI (o1, o3, GPT-5):**
- Thinking is **hidden** by default
- You see summary or nothing
- Can't see raw reasoning tokens
- `reasoning_effort: low/medium/high` controls thinking budget

**Anthropic (Claude thinking):**
- Thinking is **visible** in `<thinking>` blocks
- You see the full reasoning trace
- `budget_tokens: N` controls max thinking tokens

**Google (Gemini thinking):**
- Thinking can be visible or hidden
- `thinking_budget: N` controls tokens

**DeepSeek R1:**
- Thinking visible in `<think>` blocks
- Model naturally generates long reasoning

---

## Provider Implementations

### OpenAI: o1, o3, GPT-5

**Architecture**: Unknown (proprietary)

**Key innovation**: "Inference-time compute scaling"
- More thinking time = better answers
- Controlled via `reasoning_effort`

**Settings:**
```yaml
reasoning:
  effort: "low"      # Fast, cheap, less accurate
  effort: "medium"   # Balanced
  effort: "high"     # Slow, expensive, most accurate
  summary: "auto"    # Show reasoning summary (optional)
```

**o3 high** achieved 87.5% on ARC-AGI (but cost ~$1000/task!)

**GPT-5** integrates reasoning into the base model:
- Not a separate "reasoning mode"
- Always has some reasoning capability
- `effort` controls how much

### Anthropic: Claude Extended Thinking

**Architecture**: Same base model, different inference mode

**How it works:**
1. Model generates `<thinking>` tokens (visible to you)
2. Thinking doesn't count toward output limit (separate budget)
3. After thinking, generates normal response

**Settings:**
```yaml
thinking:
  type: "enabled"
  budget_tokens: 16000  # Max thinking tokens
```

**Key insight**: Same model weights, different generation mode. The thinking tokens are "real" output that you can see and debug.

### Google: Gemini Thinking

**Architecture**: Similar to Anthropic

**Settings:**
```yaml
thinking_config:
  thinking_budget: 8000
```

**Gemini 2.5 Pro** with thinking achieved strong results on reasoning benchmarks.

### DeepSeek: R1

**Architecture**: Mixture-of-Experts (671B total, 37B active)

**Training**:
1. Cold start with limited CoT examples
2. Large-scale RL with GRPO
3. Self-generated reasoning data
4. Distillation to smaller models

**Key innovation**: Fully open-source, including training details

**R1 naturally generates very long reasoning** (often 10K+ tokens for complex problems)

### Comparison Table

| Model | Reasoning Visible | Control | Best For |
|-------|-------------------|---------|----------|
| o1/o3 | Hidden | effort | Math, code |
| GPT-5 | Summary | effort | General |
| Claude thinking | Visible | budget_tokens | Debugging, transparency |
| Gemini thinking | Configurable | thinking_budget | Flexible |
| DeepSeek R1 | Visible | Natural | Cost-effective |

---

## Inference-Time Compute Scaling

### The Key Insight

Traditional scaling: **More training compute = smarter model**
- Train larger models
- Train on more data
- Diminishing returns (10x compute ≠ 10x smarter)

New paradigm: **More inference compute = better answers**
- Same model, more thinking time
- Linear scaling (2x thinking ≈ 2x better on hard problems)
- Pay only when you need it

### How It Works

```
Easy question: "What's 2+2?"
→ Model thinks briefly (100 tokens)
→ Cost: $0.001

Hard question: "Prove Fermat's Last Theorem"
→ Model thinks extensively (100,000 tokens)
→ Cost: $10.00
```

Same model, different compute allocation.

### The Scaling Law

OpenAI showed (roughly):
```
Performance ∝ log(thinking_tokens)
```

Doubling thinking time gives consistent improvement, but with diminishing returns.

### Implications for ARC

ARC puzzles benefit from inference-time compute:
- More thinking = more hypotheses tried
- More verification steps
- Better chance of finding correct pattern

**But**: After ~50K thinking tokens, returns diminish. Better to run multiple experts than one very-long-thinking expert.

---

## When to Use What

### Use Standard LLM (no reasoning) when:
- Task is simple (lookup, formatting, translation)
- Speed is critical
- Cost is primary concern
- Task doesn't require multi-step logic

### Use Low Reasoning Effort when:
- Moderate complexity
- Good balance of speed/accuracy needed
- Budget constrained

### Use Medium Reasoning Effort when:
- Complex but not extreme problems
- ARC-style puzzles (good default)
- Code generation with logic

### Use High Reasoning Effort when:
- Very complex problems
- Math proofs
- Critical accuracy needed
- Cost is secondary

### Use Multiple Experts (your approach) when:
- Problems have multiple valid solution paths
- Single expert might get stuck in local minimum
- Consensus improves confidence
- Diversity matters more than depth

---

## Cost-Performance Tradeoffs

### Token Economics

Reasoning models use more tokens:
```
Standard: ~500 output tokens
Reasoning (low): ~2,000 output tokens
Reasoning (medium): ~10,000 output tokens
Reasoning (high): ~50,000+ output tokens
```

### Cost Comparison (approximate)

| Model | Mode | ARC Task Cost | Expected Accuracy |
|-------|------|---------------|-------------------|
| GPT-4o | Standard | $0.05 | ~20% |
| Claude Sonnet | Standard | $0.10 | ~25% |
| GPT-5 | Low | $0.50 | ~35% |
| GPT-5 | Medium | $2.00 | ~45% |
| Claude thinking 8K | - | $1.50 | ~40% |
| GPT-5 | High | $5.00 | ~52% |
| o3 | Low | $3.00 | ~50% |
| o3 | High | $50+ | ~75%+ |
| Your hybrid | 8 experts | ~$3-5 | ~60-70%? |

### The Ensemble Advantage

Instead of:
- 1 expert × high reasoning = expensive, single point of failure

Your approach:
- 8 experts × medium reasoning = similar cost, diversity, consensus

**Why this works:**
- Different experts find different solutions
- Voting filters out lucky/unlucky runs
- More robust than single expensive run

---

## Summary

### Key Takeaways

1. **Reasoning models are trained differently** - RL with process rewards, not just next-token prediction

2. **Thinking is real computation** - not just prompted behavior, but trained capability

3. **Inference-time scaling** - more thinking = better results (up to a point)

4. **Tradeoff triangle**: Speed ↔ Cost ↔ Accuracy

5. **Ensembles beat single experts** - for ARC-style problems, diversity > depth

### For Your ARC Solver

Your current setup:
- GPT-5.2 with `reasoning_effort: "medium"`
- 8 parallel experts
- Perception-guided prompts
- Voting for consensus

This is a smart balance:
- Medium reasoning = good thinking without excessive cost
- 8 experts = diversity and consensus
- Perception = domain-specific guidance
- Voting = filters overfitting

**Potential experiments:**
- Try `"high"` on a subset to see if accuracy jumps
- Compare thinking token count vs accuracy
- Test if perception helps more with low or high reasoning

---

## Further Reading

1. **"Let's Verify Step by Step"** (OpenAI, 2023) - Process reward models
2. **"Training Verifiers to Solve Math Word Problems"** (OpenAI, 2021) - Early reasoning work
3. **"DeepSeek R1 Technical Report"** (2024) - Open reasoning model training
4. **"Scaling Laws for Reward Model Overoptimization"** (2022) - RL training insights
5. **"Constitutional AI"** (Anthropic, 2022) - RLAIF approach
6. **"Chain-of-Thought Prompting"** (Google, 2022) - Original CoT paper
