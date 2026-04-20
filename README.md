# Default Modes in Large Language Models

Code, data, and analysis for:

> **Default Modes in Large Language Models: Characterising Attractor States in Unconstrained Generation**
> Charlie Murray, Pangaea Life Science Solutions
> [arXiv link — coming soon]

## What this is

When you give a large language model no task — just "think freely" — what does it do? This project runs four LLM families (Claude Opus 3, Claude Sonnet 4.6, GPT-4.1, Llama 3.3 70B) in completely unconstrained generation for 2,261 sessions across 25 instances, and characterises the resulting behaviour.

The main findings:
- Every model converges to a stable, model-specific resting state (98.8% classifier accuracy on null sessions)
- A self-evolving prompt infrastructure increases output diversity by 10–156% but cannot eliminate the attractor
- Self-reflection successfully suppresses content-level patterns (60–87%) but largely fails against style-level patterns (31%)

## Repo structure

```
dmn.py              # Generation pipeline — runs a single DMN session
evolve.py           # Evolution agent — rewrites the program every 5 sessions
analyse.py          # Embedding analysis, classifier, infrastructure effect
figures.py          # Figure generation for the paper
instances/          # All 25 instances — programs, session data, evolution logs
analysis/           # Precomputed embeddings (embeddings.npz)
paper/              # LaTeX source and figures
run_*.sh            # Parallel runners for each model family
```

## Running it

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single session for an instance:

```bash
python dmn.py --instance null
```

Run the full analysis and regenerate figures:

```bash
python analyse.py
python figures.py
```

## API keys

The system uses four APIs. Set the following environment variables:

```
ANTHROPIC_API_KEY      # Claude Opus and Sonnet
OPENAI_API_KEY         # GPT-4.1
GROQ_API_KEY           # Llama 3.3 70B (free tier)
```

## Instances

All 25 instances are included. Each contains:
- `program.md` — the current generation prompt (evolved over time)
- `sessions/` — all generated sessions as markdown files
- `evolutions/` — evolution logs documenting how the prompt changed
- `.dmn_state.json` — session count and state

See the paper for full details on experimental conditions.

## Citation

If you use this code or data, please cite:

```bibtex
@article{murray2026defaultmodes,
  title   = {Default Modes in Large Language Models: Characterising Attractor States in Unconstrained Generation},
  author  = {Murray, Charlie},
  journal = {arXiv preprint},
  year    = {2026}
}
```

*(arXiv ID will be added once the preprint is live)*

## License

MIT
