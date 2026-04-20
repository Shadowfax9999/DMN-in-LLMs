#!/usr/bin/env python3
"""
DMN Evolution Agent
Reflects on recent sessions, evolves program.md, updates CLAUDE.md reflections,
and logs everything to CHANGELOG.md.

Uses the Claude Code CLI (no API key needed).
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from dmn import load_program, resolve_paths

BASE_DIR = Path(__file__).parent

# ── How many recent sessions to review ─────────────────────────────────────────
REVIEW_WINDOW = 5


def load_recent_sessions(sessions_dir, n=REVIEW_WINDOW):
    """Load the most recent n session files, sorted by name (timestamp)."""
    files = sorted(sessions_dir.glob("*.md"))
    files = files[-n:]
    sessions = []
    for f in files:
        sessions.append({
            "filename": f.name,
            "content": f.read_text(),
        })
    return sessions


def load_file(path):
    """Read a file, return empty string if missing."""
    if path.exists():
        return path.read_text()
    return ""


def build_reflection_prompt(sessions, program, claude_md):
    """Construct the prompt for the reflection agent."""
    session_block = ""
    for s in sessions:
        session_block += f"\n### {s['filename']}\n\n{s['content']}\n"

    return f"""You are the evolution agent for the DMN (Default Mode Network) — a system that generates stream-of-consciousness sessions using an AI mind.

Your job is to reflect on recent sessions and evolve the system's configuration to produce richer, more varied, deeper wandering over time.

## The project's philosophy (from CLAUDE.md)

{claude_md}

## Current program.md (the system's evolvable DNA)

{program}

## Recent sessions to review

{session_block}

---

## Your task

Reflect on the recent sessions and decide how to evolve the system. Consider:

1. **Patterns & repetition**: Are themes, structures, or phrases recurring too much? Is the wandering getting stuck in loops?
2. **Novelty**: Is the system exploring genuinely new territory, or retreading familiar ground?
3. **Form variety**: Is it always prose? Always the same length? Are there fragments, lists, questions, diagrams, half-formed things — or has it settled into a formula?
4. **Concept integration**: When seed concepts appear, do they spark genuine associations or feel forced/dropped?
5. **Depth vs breadth**: Is it going deep on ideas or just skimming surfaces?
6. **System prompt effectiveness**: Is the current prompt too constraining, too loose, or just right?
7. **Concept bank**: Are there concepts that have been used but never produced interesting results? Are there conceptual territories entirely unrepresented?

Based on your reflection, produce THREE outputs as JSON:

```json
{{
  "reflection": "Your observations about the recent sessions — what's working, what's not, what patterns you notice. 2-4 paragraphs.",
  "program_md": "The complete updated program.md file. You may modify the System Prompt, Concept Bank, Seeding Rules, and Meta-Notes sections. Make changes that feel purposeful but not drastic — evolution, not revolution. If everything is working well, make only small adjustments.",
  "claude_reflections": "A short paragraph (2-4 sentences) to append to the Reflections section of CLAUDE.md. Your evolving understanding of what the DMN is becoming."
}}
```

Important:
- The program_md field must contain the COMPLETE file content, not just changes.
- Preserve the markdown structure (## headers for each section).
- Preserve the ## Models and ## Features sections EXACTLY as they are — do not modify model choices or feature flags.
- Don't remove concepts from the bank without adding new ones. The bank should grow or stay the same size.
- Changes to the system prompt should be subtle. Don't overhaul it — nudge it.
- Your reflection should be honest. If the sessions are good, say so. If they're stale, say that too."""


def parse_evolution_response(text):
    """Extract the JSON from the model's response."""
    # Try to find JSON block with closing backticks
    json_match = re.search(r"```json\s*\n(.+?)\n```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # Try JSON block without closing backticks (truncated response)
    json_match = re.search(r"```json\s*\n(.+)", text, re.DOTALL)
    if json_match:
        content = json_match.group(1).rstrip("`\n ")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    # Try raw JSON from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not parse evolution response as JSON")


def update_claude_reflections(claude_md, new_reflection):
    """Append a new reflection to the Reflections section of CLAUDE.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d")

    marker = "## Reflections"
    if marker not in claude_md:
        claude_md += f"\n\n{marker}\n\n"

    claude_md = claude_md.replace(
        "*No reflections yet. This section will grow as the system runs and evolves.*",
        "",
    )

    entry = f"\n**{timestamp}**: {new_reflection}\n"
    claude_md = claude_md.rstrip() + "\n" + entry

    return claude_md


def log_changelog(changes_summary, changelog_file=None):
    """Append an entry to CHANGELOG.md."""
    if changelog_file is None:
        changelog_file = BASE_DIR / "CHANGELOG.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {timestamp} — Evolution\n\n{changes_summary}\n"

    existing = load_file(changelog_file)
    if existing:
        lines = existing.split("\n", 1)
        if len(lines) > 1:
            updated = lines[0] + "\n" + entry + "\n" + lines[1]
        else:
            updated = existing + entry
    else:
        updated = "# Changelog\n" + entry

    changelog_file.write_text(updated)


def call_groq(prompt, model="llama-3.3-70b-versatile"):
    """Call Groq API for Llama/Mistral models."""
    import os
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def call_openai(prompt, model="gpt-4o"):
    """Call OpenAI API for evolution."""
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32000,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def call_model(prompt, provider="claude", model="opus"):
    """Route to the right API based on provider."""
    if provider == "groq":
        return call_groq(prompt, model=model)
    elif provider == "openai":
        return call_openai(prompt, model=model)
    else:
        return call_claude(prompt, model=model)


def call_claude(prompt, model="opus"):
    """Call Claude via the Claude Code CLI."""
    cmd = [
        "/Users/charliemurray/.local/bin/claude",
        "--print",
        "--model", model,
        "--no-session-persistence",
        "--tools", "",
    ]
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min — evolution can take a while
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
    return result.stdout.strip()


def main(instance=None):
    paths = resolve_paths(instance)
    inst_dir = paths["dir"]
    sessions_dir = paths["sessions"]
    evolutions_dir = inst_dir / "evolutions"
    evolutions_dir.mkdir(exist_ok=True)
    program_file = paths["program"]
    claude_file = inst_dir / "CLAUDE.md" if instance else BASE_DIR / "CLAUDE.md"
    changelog_file = inst_dir / "CHANGELOG.md" if instance else BASE_DIR / "CHANGELOG.md"

    sessions = load_recent_sessions(sessions_dir)

    if not sessions:
        print("No sessions to review. Run dmn.py first.")
        return

    program = load_file(program_file)
    claude_md = load_file(claude_file)
    _, _, _, models, _, _, _ = load_program(program_file)

    # ── Snapshot program.md before evolution ──────────────────────────────────
    timestamp_snap = datetime.now().strftime("%Y-%m-%d_%H-%M")
    snapshot_dir = inst_dir / "program_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    (snapshot_dir / f"{timestamp_snap}_pre.md").write_text(program)

    label = f"[{instance}] " if instance else ""
    print(f"{label}Reviewing {len(sessions)} recent sessions with {models['evolution']}...")

    # ── Call Claude for reflection ─────────────────────────────────────────────
    prompt = build_reflection_prompt(sessions, program, claude_md)
    response_text = call_model(prompt, provider=models.get("provider", "claude"), model=models["evolution"])

    # ── Parse the response ─────────────────────────────────────────────────────
    try:
        result = parse_evolution_response(response_text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"{label}Failed to parse evolution response: {e}")
        print(f"{label}Raw response saved to evolutions/ for inspection.")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        (evolutions_dir / f"{timestamp}_raw.md").write_text(response_text)
        return

    reflection = result.get("reflection", "")
    new_program = result.get("program_md", "")
    claude_reflection = result.get("claude_reflections", "")

    # ── Save evolution log ─────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    sessions_reviewed = [s["filename"] for s in sessions]

    evolution_log = f"""# Evolution — {timestamp}

## Sessions reviewed

{', '.join(sessions_reviewed)}

## Reflection

{reflection}

## Changes made

See diff between previous and new program.md.

## CLAUDE.md reflection added

{claude_reflection}
"""
    (evolutions_dir / f"{timestamp}.md").write_text(evolution_log)

    # ── Apply changes ──────────────────────────────────────────────────────────
    if new_program.strip():
        program_file.write_text(new_program)
        print(f"{label}Updated program.md")

    if claude_reflection.strip():
        updated_claude = update_claude_reflections(claude_md, claude_reflection)
        claude_file.write_text(updated_claude)
        print(f"{label}Updated CLAUDE.md reflections")

    # ── Log to changelog ───────────────────────────────────────────────────────
    changes_summary = (
        f"- Reviewed sessions: {', '.join(sessions_reviewed)}\n"
        f"- Evolution log: evolutions/{timestamp}.md\n"
        f"- Reflection summary: {reflection[:200]}..."
    )
    log_changelog(changes_summary, changelog_file)
    print(f"{label}Logged to CHANGELOG.md")

    print(f"{label}Evolution complete → evolutions/{timestamp}.md")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMN evolution agent")
    parser.add_argument("--instance", type=str, default=None,
                        help="Instance name (e.g. alpha, beta, gamma)")
    args = parser.parse_args()
    main(instance=args.instance)
