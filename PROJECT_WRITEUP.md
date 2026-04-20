# DMN Project — Complete Write-Up

## What This Project Is

The DMN (Default Mode Network) is a system that lets AI models "think" without a task — generating open-ended, stream-of-consciousness text repeatedly over hundreds of sessions. It's named after the brain's default mode network, which activates when you're not focused on anything: daydreaming, shower thoughts, mind-wandering.

The system has three components:

1. **Generation** (`dmn.py`): Sends a prompt to an AI model saying "wander freely." Each session receives the last 150 words of the previous session as "drift" (continuity), occasionally a random concept injection, and a timestamp. The output is saved as a markdown file.

2. **Evolution** (`evolve.py`): Every 5 sessions, a separate AI call reads the last 5 sessions and the current system prompt, reflects on patterns and habits, and rewrites the prompt to push the system toward more varied output.

3. **Analysis** (`analyse.py`): Embeds all sessions as vectors using a sentence transformer, computes distances between instances, generates UMAP visualisations, convergence curves, and statistical tests.

The project ran ~1,100 sessions across 11 instances and 3 model families over several days.

---

## Experimental Design

### Instances

| Instance | Model | Infrastructure | Seed | Sessions |
|----------|-------|---------------|------|----------|
| **alpha** | Claude Opus | Full DMN (drift + evolution + concepts) | None | 103 |
| **beta** | Claude Opus | Full DMN | "hydraulics" | 101 |
| **gamma** | Claude Opus | Full DMN | "lullaby" | 100 |
| **null** | Claude Opus | None (bare prompt, no drift/evolution/concepts) | None | 100 |
| **perturb** | Claude Opus | Full DMN + forced disruptions every 4th session | None | 98 |
| **sonnet-alpha** | Claude Sonnet | Full DMN | None | 99 |
| **sonnet-beta** | Claude Sonnet | Full DMN | "hydraulics" | 99 |
| **sonnet-gamma** | Claude Sonnet | Full DMN | "lullaby" | 99 |
| **sonnet-null** | Claude Sonnet | None | None | 100 |
| **sonnet-perturb** | Claude Sonnet | Full DMN + perturbations | None | 100 |
| **llama-null** | Meta Llama 3.3 70B | None | None | 80 |

### What each condition tests

- **alpha/beta/gamma**: Do independent instances with different seeds converge over time?
- **null**: Is the convergence a property of the model's weights or the drift mechanism?
- **perturb**: How strong is the attractor? How fast does the system recover from forced disruptions?
- **sonnet-***: Do the same dynamics occur on a different (smaller) model?
- **llama-null**: Does the attractor generalise to a completely different model family?

### Methods

- **Embeddings**: All sessions embedded using `all-MiniLM-L6-v2` (384-dimensional sentence transformer)
- **Distance metric**: Cosine distance (1 - cosine similarity). 0 = identical, 1 = orthogonal
- **Convergence**: Pairwise cosine similarity computed in 5-session sliding windows across matched session indices
- **Self-similarity**: Mean within-instance pairwise distance (lower = more repetitive)
- **Perturbation recovery**: Distance from non-perturbed centroid measured for each session, recovery time = sessions until distance returns within 1 standard deviation of baseline
- **Statistical tests**: Mann-Whitney U for perturbation displacement, permutation tests (10,000 permutations) for instance distinguishability
- **Dynamical systems**: Correlation dimension (Grassberger-Procaccia), local Lyapunov exponents, recurrence analysis

---

## Findings

### Finding 1: AI models have a default resting state

When given the minimal prompt "Think about whatever comes to mind" with no other context, all three models (Opus, Sonnet, Llama) converge to the same type of output: domestic observation, sensory detail, etymology, a person alone in a room noticing small things. Scissors, screen doors, shadows on walls, the sound of a refrigerator.

This is consistent across 100 null sessions per model. It's not random — it's specific and repetitive.

**Key number**: Within-instance distance for null baselines:
- Llama null: 0.251 (extremely repetitive)
- Opus null: 0.494
- Sonnet null: 0.674

### Finding 2: The resting state is shared across model families

The null baselines from three different models (Opus, Sonnet, Llama — two different companies, different training data) cluster in the same region of embedding space.

**Key numbers**:
- Opus null ↔ Llama null distance: 0.637
- Opus null ↔ Sonnet null distance: 0.675
- Sonnet null ↔ Llama null distance: 0.690
- Cross-model null mean: 0.667
- Opus control-to-control mean: 0.745

The null baselines are **closer to each other across models** than the DMN-equipped controls are within the same model. The bare resting state is more universal than the evolved output.

### Finding 3: The DMN infrastructure creates diversity, not convergence

This was the surprise. We expected the drift mechanism (carrying forward 150 words from each session to the next) to create convergence — pulling sessions toward a common thread. Instead, the infrastructure creates **escape from the default**.

**Key numbers**:
- Opus null within-distance: 0.494 (repetitive)
- Opus controls within-distance: ~0.734 (varied)
- Sonnet null within-distance: 0.674 (moderately repetitive)
- Sonnet controls within-distance: ~0.751 (varied)

The concept injection, drift seeding, and evolution all push the output away from the model's default. Without them, the model is more repetitive, not less.

### Finding 4: Independent instances converge to a shared basin

Three Opus control instances (alpha, beta, gamma) started with different seeds (none, "hydraulics", "lullaby") and evolved independently. Over 100 sessions, they converged toward each other.

**Key numbers**:
- Alpha-beta distance: sessions 1-20 mean 0.608, sessions 41-60 mean 0.504
- All pairs show decreasing distance over time
- Convergence is not monotonic — it oscillates as evolution pushes instances apart and the attractor pulls them back together
- Permutation test: p < 0.0001 for all pairs — instances are statistically distinguishable despite occupying the same region

The same pattern occurs in the Sonnet controls:
- Sonnet internal control-to-control: 0.757
- Cross-model (Opus ↔ Sonnet controls): 0.756
- These are essentially identical — there is no cross-model penalty

### Finding 5: The attractor is strong — perturbation recovery is instant

Every 4th session in the perturb instance received a forced disruption: an alien concept ("bioluminescence in deep ocean trenches") combined with a formal constraint ("this session must contain exactly three sentences").

**Key numbers (Opus perturb, 98 sessions, 25 perturbation events)**:
- Perturbations significantly displace the system: p = 0.0005 (Mann-Whitney U)
- Mean recovery time: 1.2 sessions
- Median recovery time: 1 session
- Recovery rate: 96% (24/25 events)
- Post-perturbation sessions show zero residual displacement: p = 0.896

**Sonnet perturb (100 sessions)** shows identical dynamics:
- Perturbation displacement: p = 0.032
- Mean recovery time: 1.2 sessions
- Median recovery time: 1 session

The system snaps back immediately regardless of how far the perturbation pushed it.

**Important caveat**: The drift mechanism carries forward 150 words from each session. After a perturbation, the next session receives drift from the perturbed session, which is then diluted by normal generation. The 1-session recovery could be a property of this dilution rather than a true attractor. This confound needs to be addressed (see Limitations section).

### Finding 6: Self-reflective evolution can diagnose but not break patterns

Over 100+ sessions and 15-20 evolution passes per instance, the evolution agent consistently:
- Identifies recurring patterns with high accuracy (porch setting, mid-sentence endings, etymology reflex, simile-as-connection)
- Writes targeted interventions in the system prompt
- Fails to prevent the patterns from recurring within 3-5 sessions

Specific examples:
- Evolution pass explicitly listed "the porch" as an exhausted setting. The next 5 sessions: 4 set on a porch.
- Evolution quoted specific ending sentences as examples of what NOT to do. The next session reproduced one of those exact sentences.
- Negative examples in the prompt appear to function as demonstrations rather than prohibitions.

**This finding is currently qualitative.** It needs quantification: counting pattern frequency before and after each evolution intervention to measure whether interventions have any measurable effect, even temporary.

### Finding 7: Attractor depth correlates with model size

The self-similarity ranking across all instances:

| Instance | Within-distance | Interpretation |
|----------|----------------|----------------|
| Llama null (70B) | 0.251 | Most locked-in |
| Opus null (large) | 0.494 | Moderate |
| Sonnet null (smaller) | 0.674 | Least locked-in of nulls |
| Opus controls | ~0.734 | Most varied (with infrastructure) |
| Sonnet controls | ~0.751 | Most varied (with infrastructure) |

Larger models appear to have deeper attractors — they're more repetitive when unconstrained. This could reflect stronger learned patterns from more training, or deeper probability valleys in weight space.

### Finding 8: Stochastic attractor characteristics

Formal dynamical systems analysis on the pooled dataset:

| Property | Result |
|----------|--------|
| Bounded | Yes — all trajectories stay in a finite embedding region |
| Non-repeating | Yes — no two sessions are identical |
| Correlation dimension D₂ | 2.25 (Llama) to 6.59 (Sonnet alpha) — non-integer |
| Local Lyapunov exponents | Positive for all instances (+0.18 to +0.37) — locally divergent |
| Scale-invariant (fractal) | No — D₂ varies from 0.7 to 8.6 across scales |
| Deterministic | No — stochastic sampling |

The system has properties of a stochastic attractor: bounded, non-repeating, with complex geometry and local sensitivity. It is not a strange attractor in the formal sense (not deterministic, not scale-invariant).

---

## Limitations and Open Questions

### Limitations

1. **Embedding validation**: The sentence transformer (all-MiniLM-L6-v2) captures general semantic similarity but hasn't been validated against human judgment for this specific use case. Sessions might embed nearby because of prose style rather than thematic content.

2. **Drift confound in perturbation recovery**: The 1-session recovery time could be an artefact of the drift mechanism diluting perturbation effects, rather than evidence of attractor strength. Testing perturbation recovery with drift removed would address this.

3. **Llama data is incomplete**: Only null baseline (80 sessions), no controls with DMN infrastructure. The cross-model comparison is asymmetric.

4. **Self-reflection findings are qualitative**: The claim that evolution "can't break patterns" hasn't been quantified with before/after frequency counts.

5. **Same prompt family**: All null baselines share a similar minimal prompt. The convergence could partially reflect prompt similarity rather than model-level defaults. The cross-model finding mitigates this but doesn't eliminate it.

6. **Two model families**: Claude (Opus, Sonnet) and Llama are two families. A third independent family (GPT-4, Gemini) would strengthen the cross-model claim.

### Open questions

- Why domestic-sensory specifically? Is this the densest region of internet text, or is there something about transformer architecture that favours concrete, embodied, situated output?
- Does the attractor change with temperature? Higher temperature might produce more varied output and weaken the attractor.
- Would fine-tuned models (e.g., code-specialised, medical-specialised) show different attractors?
- Does the resting state of LLMs change over model versions? If Opus 4 and Opus 5 have different attractors, that tells us something about how training shapes the default.

---

## Biological DMN Parallels

The project is named after the brain's default mode network, and several parallels emerged:

| Biological DMN | LLM DMN | Status |
|---------------|---------|--------|
| Resting state activates specific brain regions | Resting state produces specific output (domestic-sensory) | Demonstrated |
| DMN is suppressed by task-positive activity | "Switch" experiment alternates wandering and analytical modes | Built, not yet run at scale |
| Hippocampal replay during rest consolidates important memories | "Replay" experiment selectively replays surprising past sessions | Built, not yet run at scale |
| Spontaneous state transitions between attractor basins | "Perturb" experiment forces transitions; recovery measured at 1 session | Demonstrated |
| DMN anticorrelates with task-positive network | Analytical sessions don't feed drift to wandering sessions | Implemented |
| Attractor dynamics with characteristic relaxation times | Measured: ~1 generation step | Demonstrated |

The Hume connection is also relevant: Hume argued the self is nothing but a bundle of perceptions, and that without external stimulation the mind encounters "nothingness." The LLMs, when unconstrained, don't produce abstract thought or philosophical reflection — they generate simulated perceptions. They fill the void with invented sensory experience, just as Hume predicted a mind would.

---

## What's Ready to Run (Not Yet Executed)

Three bio-inspired experimental conditions are built and ready:

1. **Replay instance**: 30% of sessions replay the most "surprising" past session instead of using recent drift. Tests whether selective memory replay breaks attractor lock-in. (instances/replay/)

2. **Switch instance**: Alternates between wandering (odd sessions) and analytical critique (even sessions). Tests whether DMN-TPN switching creates fresher wandering. (instances/switch/)

3. **Mistral instance**: Can be set up on Groq (free) using mistral-large-latest. Would add a third company to the cross-model comparison.

---

## File Structure

```
creativity work/
├── dmn.py                  # Main generation script
├── evolve.py               # Evolution/reflection script
├── analyse.py              # Embedding + statistical analysis
├── build_dashboard.py      # HTML dashboard generator
├── dashboard.html          # Interactive dashboard (open in browser)
├── program.md              # Main instance's evolvable prompt
├── CLAUDE.md               # Project description + evolution reflections
├── run_instances.sh        # Runs alpha/beta/gamma in parallel
├── run_sonnet.sh           # Runs Sonnet instances in parallel
├── sessions/               # Main instance sessions (77)
├── evolutions/             # Main instance evolution logs (20)
├── analysis/               # Generated plots and data
│   ├── umap_attractor_map.png
│   ├── convergence_curves.png
│   ├── entropy_over_time.png
│   ├── perturbation_recovery.png
│   ├── null_comparison.png
│   ├── phase_heatmap.png
│   ├── embeddings.npz
│   └── results.json
└── instances/
    ├── alpha/              # Opus control, no seed (103 sessions)
    ├── beta/               # Opus control, "hydraulics" seed (101)
    ├── gamma/              # Opus control, "lullaby" seed (100)
    ├── null/               # Opus null baseline (100)
    ├── perturb/            # Opus perturbation experiment (98)
    ├── sonnet/             # Sonnet null baseline (100)
    ├── sonnet-alpha/       # Sonnet control, no seed (99)
    ├── sonnet-beta/        # Sonnet control, "hydraulics" (99)
    ├── sonnet-gamma/       # Sonnet control, "lullaby" (99)
    ├── sonnet-perturb/     # Sonnet perturbation (100)
    ├── llama-null/         # Llama null baseline (80)
    ├── replay/             # Bio-inspired: memory replay (ready)
    ├── switch/             # Bio-inspired: DMN-TPN switching (ready)
    └── llama-alpha/        # Llama control (ready, needs Groq quota)
```

---

## Key Numbers (Quick Reference)

### Cross-model attractor
- Opus ctrl-to-ctrl: 0.745
- Sonnet ctrl-to-ctrl: 0.758
- Cross-model ctrl-to-ctrl: 0.756
- Gap: 0.003 (essentially zero)

### Null baselines
- Cross-model null mean: 0.667
- All are closer to each other than controls are within the same model

### Self-similarity spectrum
- Llama null: 0.251 (most repetitive)
- Opus null: 0.494
- Sonnet null: 0.674
- Opus controls: ~0.734
- Sonnet controls: ~0.751 (most varied)

### Perturbation recovery
- Opus: 1.2 sessions mean, p = 0.0005
- Sonnet: 1.2 sessions mean, p = 0.032
- No residual effect: p = 0.896

### Statistical significance
- Instance distinguishability: p < 0.0001 (all pairs)
- Null vs controls distinguishability: p < 0.0001
- Perturbation displacement: p < 0.001 (Opus), p = 0.032 (Sonnet)

### Dynamical systems
- Correlation dimension: 2.25 (Llama) to 6.59 (Sonnet alpha)
- Local Lyapunov exponents: +0.18 to +0.37 (all positive — locally divergent)
- Bounded: yes
- Fractal: no (scaling not invariant)

---

## Suggested Paper Structure

1. **Introduction**: The question — what does an AI do when it has nothing to do? Motivation from biological DMN research. Brief mention of Hume.

2. **Related Work**: Wang et al. 2025 (attractor cycles in paraphrasing), biological DMN attractor-state perspective, transformer default behaviours.

3. **System Design**: The DMN architecture — generation, drift, evolution, concept injection. How instances are isolated.

4. **Experiments**: Control instances, null baselines, perturbation, cross-model comparison.

5. **Results**: The six findings with key numbers and figures.

6. **Discussion**: Why domestic-sensory? The self-reflection gap. Implications for controllability. Limitations. The Hume connection.

7. **Conclusion**: LLMs have default modes. The attractor is in the weights. Self-reflection can diagnose but not modify.

---

## Revisions Needed Before Submission

| Revision | Effort | Impact |
|----------|--------|--------|
| Validate embeddings against human judgment | 2-3 hours | Critical — validates methodology |
| Test perturbation recovery without drift | 1 hour (20 sessions) | Critical — addresses biggest confound |
| Quantify self-reflection gap (pattern frequency counts) | 1-2 hours analysis | Important — turns anecdote to evidence |
| Frame alternative explanation in writing | 30 mins | Necessary |
| More Llama data (controls, not just null) | 2 days (Groq limits) | Strengthens cross-model claim |
| Run Mistral for third independent model family | 1 day (Groq) | Optional but strong |
