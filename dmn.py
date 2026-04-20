#!/usr/bin/env python3
"""
DMN — Default Mode Network
A mind wandering through its own knowledge space.
No task. No objective. Just association.

Reads its configuration from program.md (the evolvable DNA).
Uses the Claude Code CLI (no API key needed).
"""

import argparse
import json
import re
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Paths (resolved at runtime based on --instance flag) ─────────────────────
BASE_DIR = Path(__file__).parent

def resolve_paths(instance=None):
    """Return paths for sessions, state, program — either root or instance dir."""
    if instance:
        inst_dir = BASE_DIR / "instances" / instance
        inst_dir.mkdir(parents=True, exist_ok=True)
    else:
        inst_dir = BASE_DIR
    sessions_dir = inst_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    return {
        "dir": inst_dir,
        "sessions": sessions_dir,
        "state": inst_dir / ".dmn_state.json",
        "program": inst_dir / "program.md",
    }

# ── Parse program.md ──────────────────────────────────────────────────────────
def load_program(program_file=None):
    """Read the evolvable DNA from program.md."""
    if program_file is None:
        program_file = BASE_DIR / "program.md"
    text = program_file.read_text()

    # Extract system prompt (between ## System Prompt and the next ##)
    system_match = re.search(
        r"## System Prompt\n\n(.+?)(?=\n## )", text, re.DOTALL
    )
    system_prompt = system_match.group(1).strip() if system_match else ""

    # Extract concept bank (comma-separated line after ## Concept Bank)
    concepts_match = re.search(
        r"## Concept Bank\n\n(.+?)(?=\n## )", text, re.DOTALL
    )
    concepts = []
    if concepts_match:
        raw = concepts_match.group(1).strip()
        concepts = [c.strip() for c in raw.split(",") if c.strip()]

    # Extract seeding rules
    rules = {
        "drift_length": 150,
        "injection_probability": 0.25,
        "injection_interval": 4,
        "recent_memory": 6,
    }
    rules_match = re.search(
        r"## Seeding Rules\n\n(.+?)(?=\n## )", text, re.DOTALL
    )
    if rules_match:
        rules_text = rules_match.group(1)
        m = re.search(r"last ~?(\d+) words", rules_text)
        if m:
            rules["drift_length"] = int(m.group(1))
        m = re.search(r"(\d+)%", rules_text)
        if m:
            rules["injection_probability"] = int(m.group(1)) / 100
        m = re.search(r"Every (\d+)(?:th|st|nd|rd) session", rules_text)
        if m:
            rules["injection_interval"] = int(m.group(1))
        m = re.search(r"last (\d+) concepts", rules_text)
        if m:
            rules["recent_memory"] = int(m.group(1))

    # Extract exhausted themes
    exhausted = []
    exhausted_match = re.search(r"\*\*Exhausted themes\*\*:.*?actively avoided.*?:\s*\*(.+?)\*", text)
    if exhausted_match:
        exhausted = [t.strip() for t in exhausted_match.group(1).split(",") if t.strip()]
    rules["exhausted_themes"] = exhausted

    # Extract models and provider (uncommented lines in ## Models section)
    models = {"generation": "sonnet", "evolution": "opus", "provider": "claude"}
    models_match = re.search(
        r"## Models\n\n(.+?)(?=\n## )", text, re.DOTALL
    )
    if models_match:
        models_text = models_match.group(1)
        for line in models_text.split("\n"):
            line = line.strip()
            if line.startswith("<!--") or line.startswith("-->"):
                continue
            # Provider line
            pm = re.match(r"^provider:\s*(.+)$", line)
            if pm:
                models["provider"] = pm.group(1).strip()
                continue
            m = re.match(r"^(generation|evolution):\s*(.+)$", line)
            if m:
                # Map full model names to CLI aliases for Claude
                raw_model = m.group(2).strip()
                if models.get("provider", "claude") == "claude":
                    if "sonnet" in raw_model:
                        models[m.group(1)] = "sonnet"
                    elif "opus" in raw_model:
                        models[m.group(1)] = "opus"
                    elif "haiku" in raw_model:
                        models[m.group(1)] = "haiku"
                    else:
                        models[m.group(1)] = raw_model
                else:
                    # For non-Claude providers, use the model name as-is
                    models[m.group(1)] = raw_model

    # Extract features (experimental flags)
    features = {
        "replay": False,
        "perturb": False,
        "switch": False,
    }
    features_match = re.search(
        r"## Features\n\n(.+?)(?=\n## |\Z)", text, re.DOTALL
    )
    if features_match:
        for line in features_match.group(1).split("\n"):
            line = line.strip()
            m = re.match(r"^(\w+):\s*(true|false)$", line, re.IGNORECASE)
            if m:
                features[m.group(1)] = m.group(2).lower() == "true"

    # Extract perturbation bank (if perturb feature is active)
    perturbation_bank = []
    perturb_match = re.search(
        r"## Perturbation Bank\n\n(.+?)(?=\n## |\Z)", text, re.DOTALL
    )
    if perturb_match:
        for line in perturb_match.group(1).strip().split("\n"):
            line = line.strip().lstrip("- ")
            if "|" in line:
                concept, form = line.split("|", 1)
                perturbation_bank.append({
                    "concept": concept.strip(),
                    "form": form.strip(),
                })

    # Extract task-positive prompts (if switch feature is active)
    task_prompts = []
    task_match = re.search(
        r"## Task-Positive Prompts\n\n(.+?)(?=\n## |\Z)", text, re.DOTALL
    )
    if task_match:
        for line in task_match.group(1).strip().split("\n"):
            line = line.strip().lstrip("- ")
            if line:
                task_prompts.append(line)

    return system_prompt, concepts, rules, models, features, perturbation_bank, task_prompts

# ── State ──────────────────────────────────────────────────────────────────────
def load_state(state_file=None):
    if state_file is None:
        state_file = BASE_DIR / ".dmn_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {
        "session_count": 0,
        "last_excerpt": None,
        "last_concepts": [],
    }

def save_state(state, state_file=None):
    if state_file is None:
        state_file = BASE_DIR / ".dmn_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

# ── Time / season ──────────────────────────────────────────────────────────────
def get_time_context():
    now = datetime.now()
    month = now.month
    hour = now.hour

    season = (
        "Winter" if month in (12, 1, 2) else
        "Spring" if month in (3, 4, 5) else
        "Summer" if month in (6, 7, 8) else
        "Autumn"
    )

    period = (
        "deep night" if hour < 5 else
        "early morning" if hour < 8 else
        "morning" if hour < 12 else
        "afternoon" if hour < 17 else
        "evening" if hour < 21 else
        "night"
    )

    return now, season, period

# ── Seed construction ──────────────────────────────────────────────────────────
def build_seed(state, concepts, rules):
    now, season, period = get_time_context()
    day = now.strftime("%A")
    time_str = now.strftime("%H:%M")

    parts = [f"[{day}, {time_str} — {season} {period}]"]

    # Drift from previous output
    if state.get("last_excerpt"):
        parts.append(f"\nWhere you left off:\n\n{state['last_excerpt']}")

    # Random concept injection
    interval = rules.get("injection_interval", 4)
    prob = rules.get("injection_probability", 0.25)
    inject = (
        random.random() < prob
        or state["session_count"] % interval == interval - 1
    )

    concept = None
    if inject and concepts:
        recent_memory = rules.get("recent_memory", 6)
        recent = set(state.get("last_concepts", [])[-recent_memory:])
        exhausted = set(rules.get("exhausted_themes", []))
        pool = [c for c in concepts if c not in recent and c not in exhausted]
        if not pool:
            pool = [c for c in concepts if c not in exhausted] or concepts
        concept = random.choice(pool)
        parts.append(f"\nA word arrives from nowhere: {concept}")

    parts.append("\nBegin.")
    return "\n".join(parts), concept

# ── Groq API call ─────────────────────────────────────────────────────────────
def call_groq(prompt, system_prompt, model="llama-3.3-70b-versatile"):
    """Call Groq API for Llama/Mistral models."""
    import os
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()

# ── Model dispatch ────────────────────────────────────────────────────────────
def call_openai(prompt, system_prompt, model="gpt-4o"):
    """Call OpenAI API for GPT models."""
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()

def call_together(prompt, system_prompt, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    """Call Together.ai API."""
    import os
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()

def call_model(prompt, system_prompt, provider="claude", model="sonnet"):
    """Route to the right API based on provider."""
    if provider == "groq":
        return call_groq(prompt, system_prompt, model=model)
    elif provider == "openai":
        return call_openai(prompt, system_prompt, model=model)
    elif provider == "together":
        return call_together(prompt, system_prompt, model=model)
    else:
        return call_claude(prompt, system_prompt, model=model)

# ── Claude Code CLI call ──────────────────────────────────────────────────────
def call_claude(prompt, system_prompt, model="sonnet"):
    """Call Claude via the Claude Code CLI."""
    import tempfile, os
    # Write system prompt to a temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(system_prompt)
        sp_path = f.name
    try:
        cmd = [
            "/Users/charliemurray/.local/bin/claude",
            "--print",
            "--model", model,
            "--system-prompt-file", sp_path,
            "--no-session-persistence",
            "--tools", "",
        ]
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Claude CLI failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[:500]}\n"
                f"stderr: {result.stderr[:500]}"
            )
        return result.stdout.strip()
    finally:
        os.unlink(sp_path)

# ── Main ───────────────────────────────────────────────────────────────────────
def compute_surprise_score(session_text, sessions_dir):
    """Compute how different a session is from prior sessions using word overlap."""
    words = set(session_text.lower().split())
    # Load last 20 sessions and compute average vocabulary
    session_files = sorted(sessions_dir.glob("*.md"))[-20:]
    if not session_files:
        return 0.5
    all_words = set()
    for f in session_files:
        text = f.read_text()
        # Skip header
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "---":
                text = "\n".join(lines[i+1:])
                break
        all_words.update(text.lower().split())
    if not all_words:
        return 0.5
    overlap = len(words & all_words) / max(len(words), 1)
    return 1.0 - overlap  # High score = low overlap = surprising


def select_replay_session(state, sessions_dir, drift_length=150):
    """Select the most surprising past session for replay instead of most recent."""
    surprise_scores = state.get("surprise_scores", {})
    if not surprise_scores:
        return None
    # Pick the session with highest surprise score
    best_file = max(surprise_scores, key=surprise_scores.get)
    best_path = sessions_dir / best_file
    if not best_path.exists():
        return None
    text = best_path.read_text()
    # Skip header
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "---":
            text = "\n".join(lines[i+1:])
            break
    words = text.strip().split()
    return " ".join(words[-drift_length:] if len(words) > drift_length else words)


def main(instance=None):
    paths = resolve_paths(instance)
    state = load_state(paths["state"])
    system_prompt, concepts, rules, models, features, perturbation_bank, task_prompts = \
        load_program(paths["program"])

    session_num = state["session_count"] + 1
    label = f"[{instance}] " if instance else ""

    # ── Stop at 100 sessions ─────────────────────────────────────────────────
    max_sessions = 100
    if session_num > max_sessions:
        print(f"{label}Already at {session_num - 1} sessions (max {max_sessions}). Skipping.")
        return

    # ── Feature: NULL (no drift, no concept, no evolution) ──────────────────
    is_null = features.get("null", False)
    is_task_positive = False
    is_perturb_session = False

    if is_null:
        now, season, period = get_time_context()
        seed_text = f"[{now.strftime('%A')}, {now.strftime('%H:%M')} — {season} {period}]\n\nBegin."
        concept = None
        print(f"{label}Generating NULL session #{session_num} with {models['generation']}...")

    # ── Feature: SWITCH (task-positive mode on even sessions) ─────────────────
    elif features.get("switch") and session_num % 2 == 0 and task_prompts:
        is_task_positive = True
        # Use the previous wandering session as input for analysis
        task_prompt_template = random.choice(task_prompts)
        prev_excerpt = state.get("last_excerpt", "")
        if prev_excerpt:
            task_seed = f"{task_prompt_template}\n\nHere is the text to work with:\n\n{prev_excerpt}"
        else:
            task_seed = task_prompt_template
        seed_text = task_seed
        concept = None
        print(f"{label}Generating TASK-POSITIVE session #{session_num} with {models['generation']}...")
    else:
        # ── Feature: REPLAY (selective memory replay) ─────────────────────────
        if features.get("replay") and random.random() < 0.3 and state.get("surprise_scores"):
            replay_excerpt = select_replay_session(state, paths["sessions"],
                                                    rules.get("drift_length", 150))
            if replay_excerpt:
                # Temporarily replace the drift excerpt with the replayed one
                original_excerpt = state.get("last_excerpt")
                state["last_excerpt"] = replay_excerpt
                print(f"{label}REPLAY: Replaying a surprising past session...")

        # ── Feature: PERTURB (strong disruption every 4th session) ────────────
        is_perturb_session = features.get("perturb") and session_num % 4 == 0 and perturbation_bank
        if is_perturb_session:
            perturbation = random.choice(perturbation_bank)
            # Override concept injection with perturbation
            seed_text, _ = build_seed(state, concepts, rules)
            # Replace the concept line and add form constraint
            seed_text += f"\n\nFORCED CONCEPT: {perturbation['concept']}"
            seed_text += f"\nFORCED FORM: {perturbation['form']}"
            concept = perturbation["concept"]
            print(f"{label}PERTURB: Injecting '{concept}' with form '{perturbation['form']}'...")
        else:
            # Feature: PERTURB_NO_DRIFT — suppress drift for 3 sessions after perturbation
            drift_suppressed = state.get("drift_suppressed_until", 0)
            if features.get("perturb_no_drift") and session_num <= drift_suppressed:
                # Clear drift — session starts fresh like a null session but with concepts
                state["last_excerpt"] = None
                print(f"{label}DRIFT SUPPRESSED (post-perturbation, until session {drift_suppressed})")
            seed_text, concept = build_seed(state, concepts, rules)

        print(f"{label}Generating session #{session_num} with {models['generation']}...")

    session_text = call_model(seed_text, system_prompt,
                              provider=models.get("provider", "claude"),
                              model=models["generation"])

    # ── Save session file ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_file = paths["sessions"] / f"{timestamp}.md"

    seed_note = "*seed: time"
    if is_task_positive:
        seed_note = "*seed: task-positive"
    elif state.get("last_excerpt"):
        seed_note += " · drift"
    if concept:
        seed_note += f" · {concept}"
    seed_note += "*"

    with open(session_file, "w") as f:
        f.write(f"# {session_num}\n\n")
        f.write(f"{seed_note}\n\n")
        f.write("---\n\n")
        f.write(session_text)
        f.write("\n")

    # ── Update state ───────────────────────────────────────────────────────────
    drift_length = rules.get("drift_length", 150)
    words = session_text.split()
    excerpt = " ".join(words[-drift_length:] if len(words) > drift_length else words)

    recent_memory = rules.get("recent_memory", 6)
    recent_concepts = state.get("last_concepts", [])
    if concept:
        recent_concepts = (recent_concepts + [concept])[-recent_memory:]

    # Feature: PERTURB_NO_DRIFT — set drift suppression window after perturbation
    if features.get("perturb_no_drift") and is_perturb_session:
        state["drift_suppressed_until"] = session_num + 3
        print(f"{label}Drift will be suppressed for sessions {session_num+1}–{session_num+3}")

    # For switch mode: only update drift from wandering sessions, not task-positive
    if is_task_positive:
        state.update({
            "session_count": session_num,
            "last_concepts": recent_concepts,
            # Don't update last_excerpt — preserve the wandering session's drift
        })
    else:
        state.update({
            "session_count": session_num,
            "last_excerpt": excerpt,
            "last_concepts": recent_concepts,
        })

    # Feature: REPLAY — update surprise scores
    if features.get("replay") and not is_task_positive:
        surprise = compute_surprise_score(session_text, paths["sessions"])
        scores = state.get("surprise_scores", {})
        scores[session_file.name] = round(surprise, 4)
        # Keep only last 50 scores
        if len(scores) > 50:
            sorted_keys = sorted(scores.keys())
            scores = {k: scores[k] for k in sorted_keys[-50:]}
        state["surprise_scores"] = scores

    save_state(state, paths["state"])

    print(f"{label}#{session_num} → sessions/{timestamp}.md")

    # ── Regenerate dashboard ──────────────────────────────────────────────────
    try:
        subprocess.run([sys.executable, str(BASE_DIR / "build_dashboard.py")],
                       capture_output=True, timeout=30)
        print(f"{label}Dashboard updated.")
    except Exception:
        pass  # non-critical

    # ── Trigger evolution periodically (skip for null baseline) ─────────────
    evolve_every = 5
    skip_evolution = is_null or features.get("no_evolution", False)
    if not skip_evolution and session_num % evolve_every == 0 and session_num > 0:
        print(f"{label}Session #{session_num} — triggering evolution...")
        try:
            import evolve as evolve_mod
            evolve_mod.main(instance=instance)
        except Exception as e:
            print(f"{label}Evolution failed: {e}")

        # Re-run analysis after evolution
        try:
            print(f"{label}Updating analysis...")
            subprocess.run([sys.executable, str(BASE_DIR / "analyse.py")],
                           capture_output=True, timeout=120)
            subprocess.run([sys.executable, str(BASE_DIR / "build_dashboard.py")],
                           capture_output=True, timeout=30)
            print(f"{label}Analysis + dashboard updated.")
        except Exception:
            pass  # non-critical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMN session generator")
    parser.add_argument("--instance", type=str, default=None,
                        help="Instance name (e.g. alpha, beta, gamma)")
    args = parser.parse_args()
    main(instance=args.instance)
