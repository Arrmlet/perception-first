# Deep Research: Perception - From Biology to AI

## How Humans See and What AI is Missing

---

## 1. Biological Vision: The Human Visual System

### 1.1 The Visual Pathway

Human vision occupies an astonishing **30-40% of the cerebral cortex's total surface area**. Visual information flows through a hierarchical processing pipeline:

```
Retina → LGN (Thalamus) → V1 → V2 → V4 → IT (Inferotemporal)
                                           ↓
                                    Object Recognition
```

| Area | Function | Complexity |
|------|----------|------------|
| **V1** | Edge detection, orientation, motion | Simple features |
| **V2** | Contours, illusory edges, border ownership | Early grouping |
| **V4** | Angles, curvatures, color | Shape fragments |
| **IT** | Whole objects, faces, categories | Object identity |

**Key Insight**: The brain doesn't just detect features—it assigns **meaning** at every level. Even V2 (very early) computes "border ownership"—which side of an edge belongs to the object.

Sources:
- [How does the brain solve visual object recognition? - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3306444/)
- [Visual cortical processing—From image to object representation](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2023.1136987/full)

---

### 1.2 Two Visual Streams: What vs How

The brain splits visual processing into two parallel pathways:

| Stream | Path | Function | Damage Causes |
|--------|------|----------|---------------|
| **Ventral** ("What") | V1 → V2 → V4 → IT | Object recognition | Can't recognize objects |
| **Dorsal** ("How/Where") | V1 → V2 → MT → Parietal | Spatial awareness, action | Can't reach for objects |

**Critical Finding**: These streams are NOT independent. The dorsal stream needs to know WHAT an object is to know HOW to interact with it.

> "The dorsal stream never acts alone, nor does the ventral stream. Both streams need to work together in an integrated way."

**For ARC**: This maps to our kitchen metaphor:
- **Ventral** = What objects are in the grid (perception)
- **Dorsal** = What actions are possible (affordances)

Sources:
- [Two-streams hypothesis - Wikipedia](https://en.wikipedia.org/wiki/Two-streams_hypothesis)
- [What and where pathways - Scholarpedia](http://www.scholarpedia.org/article/What_and_where_pathways)

---

### 1.3 Gestalt Principles: The Brain's Grouping Rules

In the 1920s, Gestalt psychologists discovered that the brain automatically groups visual elements using specific rules:

| Principle | Rule | Example |
|-----------|------|---------|
| **Proximity** | Close things group together | ●● ●● → two pairs |
| **Similarity** | Similar things group together | ●●○○ → two groups |
| **Continuity** | We see smooth, continuous lines | Crossing lines stay separate |
| **Closure** | We complete incomplete shapes | ◗ → we see a circle |
| **Figure-Ground** | We separate object from background | Vase/faces illusion |
| **Prägnanz** | We see the simplest interpretation | "Law of simplicity" |

> "The whole is something else than the sum of its parts."

**For ARC**: The brain AUTOMATICALLY does what we're trying to teach LLMs:
- Group pixels into objects (proximity, similarity)
- See patterns (continuity)
- Complete missing parts (closure)
- Find the simplest rule (prägnanz)

Sources:
- [A Century of Gestalt Psychology in Visual Perception - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3482144/)
- [Gestalt Principles of Perception](https://courses.lumenlearning.com/waymaker-psychology/chapter/gestalt-principles-of-perception/)

---

### 1.4 Predictive Coding: The Brain as Prediction Machine

Karl Friston's **Free Energy Principle** revolutionized our understanding of perception:

> "The brain is not a passive receiver of sensory data—it actively PREDICTS what it will see, then updates based on prediction errors."

```
         ┌─────────────────────────────────────────┐
         │         HIGHER CORTICAL AREAS           │
         │                                         │
         │    Generate PREDICTIONS about input     │
         │              ↓ (top-down)               │
         ├─────────────────────────────────────────┤
         │         LOWER CORTICAL AREAS            │
         │                                         │
         │    Compute PREDICTION ERROR             │
         │              ↑ (bottom-up)              │
         ├─────────────────────────────────────────┤
         │           SENSORY INPUT                 │
         └─────────────────────────────────────────┘
```

**Key Points**:
1. Top-down signals PREDICT what lower areas should see
2. Bottom-up signals carry PREDICTION ERRORS (surprises)
3. Perception = minimize prediction error
4. Learning = update internal model to predict better

**For ARC**: This is exactly what iterative solving does:
- Hypothesis = prediction of rule
- Verification = compute prediction error
- Feedback = update the model

Sources:
- [Predictive coding under the free-energy principle - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- [The free-energy principle: a unified brain theory? - Nature](https://www.nature.com/articles/nrn2787)

---

### 1.5 Selective Attention: The Spotlight

The brain can't process everything—it uses **selective attention** to focus:

> "Selective visual attention describes the tendency of visual processing to be confined largely to stimuli that are relevant to behavior."

**Spotlight Model**:
- Attention acts like a spotlight, enhancing processing at attended locations
- fMRI shows attention literally increases activity in corresponding V1 regions
- Unattended stimuli are suppressed

**Competition Model**:
- Objects in the visual field COMPETE for processing
- Bottom-up: salient things grab attention (bright, moving, unique)
- Top-down: goals direct attention (looking for red, looking for faces)

**For ARC**: This is why perception_v10 failed:
- 400+ objects competing for attention
- LLM didn't know which to attend to
- Noise overwhelmed signal

Sources:
- [On the role of selective attention in visual perception - PNAS](https://www.pnas.org/doi/10.1073/pnas.95.3.825)
- [A physiological correlate of the 'spotlight' of visual attention - Nature](https://www.nature.com/articles/nn0499_370)

---

## 2. Gibson's Affordance Theory

### 2.1 What Are Affordances?

James J. Gibson (1979) proposed a radical idea:

> "The affordances of the environment are what it offers the animal, what it provides or furnishes, either for good or ill."

**Key Properties**:
1. Affordances are **relational** - between organism and environment
2. They are **directly perceived** - not computed through reasoning
3. They are **action possibilities** - what you CAN DO with something

| Object | Affordances |
|--------|-------------|
| Chair | Sit-on-able |
| Ladder | Climb-on-able |
| Apple | Grasp-able, eat-able |
| Knife | Cut-with-able |

**Gibson's Radical Claim**: We don't see objects then infer actions. We DIRECTLY PERCEIVE what actions are possible.

> "An affordance is neither an objective property nor a subjective property; or it is both. It is equally a fact of the environment and a fact of behavior."

Sources:
- [Gibson's Theory of Affordances - Brown CS](https://cs.brown.edu/courses/cs137/2017/readings/Gibson-AFF.pdf)
- [Affordance - Wikipedia](https://en.wikipedia.org/wiki/Affordance)

---

### 2.2 Affordances in Design

Donald Norman adapted Gibson's theory for human-computer interaction:

| Element | Affordance |
|---------|------------|
| Button | Push-able |
| Handle | Pull-able |
| Slider | Slide-able |
| Text field | Type-into-able |

**Good design makes affordances visible**. You shouldn't need instructions—the object tells you what it does.

---

### 2.3 Affordances for ARC

This is your **kitchen metaphor**:

| Grid Element | State | Affordance |
|--------------|-------|------------|
| Color-8 rectangle | MARKER | Reference point for extraction |
| Full-span line | DIVIDER | Split grid here |
| Output-sized region | EXTRACTABLE | Can be cropped |
| Pattern with anomalies | REPAIRABLE | Fix by majority |
| Empty region | FILLABLE | Destination for pattern |

**The Key Insight**:

> LLMs fail on ARC not because they can't reason, but because they don't know which actions are LEGAL at each step.

Humans look at a grid and IMMEDIATELY perceive:
- "That rectangle can be extracted"
- "That line divides the grid"
- "That pattern needs repair"

LLMs see pixels and try to COMPUTE the transformation, missing the direct perception of action possibilities.

---

## 3. AI Vision vs Human Vision

### 3.1 How Deep Learning "Sees"

| Layer | What it detects |
|-------|-----------------|
| Conv1 | Edges, blobs, colors |
| Conv2-3 | Textures, patterns |
| Conv4-5 | Parts (eyes, wheels) |
| FC | Objects, categories |

This LOOKS like the visual hierarchy (V1→V2→V4→IT), but there are critical differences.

### 3.2 Key Differences

| Aspect | Human Vision | Deep Learning |
|--------|--------------|---------------|
| **Training** | Few examples | Millions of images |
| **Generalization** | Abstract rules | Statistical patterns |
| **Robustness** | Handles novel transforms | Fails on OOD |
| **Grouping** | Automatic (Gestalt) | Not built-in |
| **Affordances** | Direct perception | Not represented |
| **Top-down** | Strong predictions | Weak/none |
| **Attention** | Dynamic, goal-driven | Fixed (mostly) |

### 3.3 The Core Problem

> "Deep neural networks are highly valuable scientific tools but should only be regarded as promising—but not yet adequate—computational models of human core object recognition behavior."

**Why DNNs fail on ARC**:
1. **No few-shot learning** - Need millions of examples
2. **No abstract rules** - Learn statistical correlations
3. **No compositionality** - Can't combine primitives
4. **No affordances** - Don't perceive action possibilities

Sources:
- [Are Deep Neural Networks Adequate Behavioral Models of Human Visual Perception?](https://www.annualreviews.org/content/journals/10.1146/annurev-vision-120522-031739)
- [Computer vision: Why it's hard to compare AI and human perception](https://bdtechtalks.com/2020/08/10/computer-vision-deep-learning-vs-human-perception/)

---

## 4. What This Means for ARC Solving

### 4.1 What Humans Do (That LLMs Don't)

| Human Process | ARC Equivalent | LLM Status |
|---------------|----------------|------------|
| Gestalt grouping | See objects, not pixels | Partial (raw grid) |
| Figure-ground | Separate objects from background | Partial |
| Affordance perception | Know what actions are possible | **MISSING** |
| Predictive coding | Hypothesize → verify → update | Yes (iterative) |
| Selective attention | Focus on relevant parts | **MISSING** |
| Prägnanz | Find simplest rule | Partial (MDL prompt) |

### 4.2 What Perception Should Provide

Based on biology, perception should give the LLM:

1. **Object segmentation** (Gestalt grouping)
   - Not 400 objects, but 2-5 meaningful ones
   - Figure-ground separation

2. **Affordance labels** (Gibson)
   - What each object CAN DO
   - What actions are LEGAL

3. **Attention guidance** (Spotlight)
   - What to focus on
   - What to ignore

4. **Pattern hints** (Predictive coding)
   - Likely rule families
   - Hypothesis space reduction

### 4.3 Why perception_v10 Failed

| Problem | Cause | Solution |
|---------|-------|----------|
| Too many objects (400+) | No attention filtering | Show only key objects |
| Verbose labels | Equal weight to all | Prioritize by relevance |
| Token dilution | Noise overwhelms signal | Compress to essentials |
| SOURCE FOUND obsession | Only helps 3.3% | Drop it |

### 4.4 What perception_slim Does Right

| Feature | Biological Basis | Tokens |
|---------|------------------|--------|
| Task type | Prägnanz (simplest interpretation) | ~20 |
| Diff map | Visual attention focus | ~100 |
| Key transitions | Change detection | ~30 |
| One affordance | Action possibility | ~20 |

**Total: ~150 tokens** with signal, not noise.

---

## 5. The Hybrid Approach: Biology + AI

### 5.1 What We Should Combine

| Component | From | Purpose |
|-----------|------|---------|
| **Perception** | Biology (Gestalt, Affordances) | Structure the problem |
| **Diversity** | AI (Ensemble) | Explore hypothesis space |
| **Reasoning** | LLM (Extended thinking) | Derive the rule |
| **Verification** | Predictive coding | Check predictions |

### 5.2 The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID SOLVER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │  PERCEPTION     │  ← Gestalt grouping                       │
│  │  (Slim)         │  ← Affordance labels                      │
│  │                 │  ← Attention guidance                      │
│  │  ~150 tokens    │  ← Pattern hints                          │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              8 PARALLEL EXPERTS                          │   │
│  │                                                          │   │
│  │   Each with different seed (diversity)                   │   │
│  │   Each with extended thinking (reasoning)                │   │
│  │   Each iterating 10 times (predictive coding)           │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  VOTING         │  ← Competition (attention)                │
│  │                 │  ← Best hypothesis wins                    │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Why This Should Work

| Biological Principle | Implementation | Expected Benefit |
|---------------------|----------------|------------------|
| Gestalt grouping | Slim perception | Better object focus |
| Affordances | Task type + hint | Know legal actions |
| Attention | Key objects only | Less noise |
| Predictive coding | Iteration + feedback | Rule refinement |
| Competition | 8 experts + voting | Best hypothesis wins |
| Prägnanz | Simplest rule prompt | Avoid overfit |

---

## 6. Conclusions

### 6.1 Key Insights from Biology

1. **Vision is hierarchical** - but meaning is assigned at every level
2. **What and How are parallel** - you need both for action
3. **Grouping is automatic** - Gestalt principles are built-in
4. **Affordances are direct** - we see action possibilities, not compute them
5. **Attention is selective** - we focus on what matters
6. **Perception is prediction** - we hypothesize and verify

### 6.2 What LLMs Are Missing

1. **Built-in grouping** - they see tokens, not objects
2. **Affordance perception** - they don't know legal actions
3. **Selective attention** - they process everything equally
4. **Few-shot generalization** - they need many examples

### 6.3 How Perception Helps

| Without Perception | With Perception |
|-------------------|-----------------|
| Raw pixels | Grouped objects |
| All details equal | Key features highlighted |
| Unknown actions | Legal actions suggested |
| Compute everything | Focus on relevant |

### 6.4 The Formula

```
Success = Perception (structure) + Diversity (exploration) + Reasoning (depth)
```

- **Perception alone**: 30% (can't explore enough)
- **Diversity alone**: 55% (no structure)
- **Perception + Diversity**: ??? (untested hypothesis)

---

## References

### Biological Vision
- [Understanding how visual information is processed in the brain - NIH](https://www.nih.gov/news-events/nih-research-matters/understanding-how-visual-information-processed-brain)
- [Visual Cognitive Neuroscience - MIT](https://oecs.mit.edu/pub/8w58nrk1/release/1)
- [How does the brain solve visual object recognition? - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3306444/)

### Gestalt Psychology
- [A Century of Gestalt Psychology in Visual Perception - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3482144/)
- [Gestalt Principles - Scholarpedia](http://www.scholarpedia.org/article/Gestalt_principles)

### Affordance Theory
- [Gibson's Ecological Approach - Brown CS PDF](https://cs.brown.edu/courses/cs137/2017/readings/Gibson-AFF.pdf)
- [Affordance - Wikipedia](https://en.wikipedia.org/wiki/Affordance)
- [The History and Philosophy of Ecological Psychology - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6280920/)

### Predictive Coding
- [Predictive coding under the free-energy principle - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- [The free-energy principle: a unified brain theory? - Nature](https://www.nature.com/articles/nrn2787)

### Visual Pathways
- [Two-streams hypothesis - Wikipedia](https://en.wikipedia.org/wiki/Two-streams_hypothesis)
- [What and where pathways - Scholarpedia](http://www.scholarpedia.org/article/What_and_where_pathways)

### Attention
- [On the role of selective attention in visual perception - PNAS](https://www.pnas.org/doi/10.1073/pnas.95.3.825)
- [Neural Mechanisms of Selective Visual Attention - PubMed](https://pubmed.ncbi.nlm.nih.gov/28051934/)

### AI Vision
- [Are Deep Neural Networks Adequate Behavioral Models? - Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev-vision-120522-031739)
- [Computer Vision Trends 2025 - Viso.ai](https://viso.ai/computer-vision/computer-vision-trends/)
