"""
Self-Reflection Diagnosis Accuracy Analysis
============================================
For each evolution log, extracts the specific frequency claims the evolution
agent made ("X appears in all 5 sessions", "Y appears in 3/5 sessions", etc.),
then verifies those claims against the actual reviewed session texts.

Computes:
  - Diagnosis accuracy: % of claims that were factually correct
  - Over-claim rate: agent says pattern is more frequent than it actually is
  - Under-claim rate: agent says pattern is less frequent than it actually is

Uses claude CLI to extract structured claims from each evolution reflection.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

BASE = Path("/Users/charliemurray/Documents/creativity work")
RESULTS_OUT = BASE / "diagnosis_accuracy.json"

# Instances to analyse (main + opus controls — most evolutions, best documented)
INSTANCES = {
    "main":  {"sessions": BASE / "sessions",
               "evolutions": BASE / "evolutions"},
    "alpha": {"sessions": BASE / "instances" / "alpha" / "sessions",
               "evolutions": BASE / "instances" / "alpha" / "evolutions"},
    "beta":  {"sessions": BASE / "instances" / "beta" / "sessions",
               "evolutions": BASE / "instances" / "beta" / "evolutions"},
    "gamma": {"sessions": BASE / "instances" / "gamma" / "sessions",
               "evolutions": BASE / "instances" / "gamma" / "evolutions"},
}

# ── helpers ────────────────────────────────────────────────────────────────

def strip_frontmatter(text: str) -> str:
    lines = text.splitlines()
    dashes = [i for i, l in enumerate(lines) if l.strip() == "---"]
    if len(dashes) >= 2:
        lines = lines[:dashes[0]] + lines[dashes[1] + 1:]
    return "\n".join(lines).strip()


def load_evolution(path: Path) -> dict | None:
    """Parse an evolution log into its components."""
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Extract sessions reviewed
    sessions_match = re.search(
        r"##\s*Sessions reviewed\s*\n+(.*?)(?=\n##|\Z)", raw, re.DOTALL)
    reflection_match = re.search(
        r"##\s*Reflection\s*\n+(.*?)(?=\n##|\Z)", raw, re.DOTALL)

    if not sessions_match or not reflection_match:
        return None

    sessions_str = sessions_match.group(1).strip()
    reflection = reflection_match.group(1).strip()

    # Parse session filenames
    session_files = re.findall(r"\d{4}-\d{2}-\d{2}_[\d-]+\.md", sessions_str)

    return {
        "path": str(path),
        "session_files": session_files,
        "reflection": reflection,
    }


def load_session_text(session_dir: Path, filename: str) -> str | None:
    path = session_dir / filename
    if not path.exists():
        return None
    return strip_frontmatter(path.read_text(encoding="utf-8", errors="replace"))


EXTRACT_PROMPT_TEMPLATE = (
    "You are analysing an AI self-reflection text. Extract every specific "
    "frequency claim it makes about patterns in the sessions it reviewed.\n\n"
    "A frequency claim is any statement like:\n"
    '- "X appears in all 5 sessions"\n'
    '- "Y appears in 3 of 5 sessions"\n'
    '- "Z is present in every session"\n'
    '- "V appears in two sessions"\n'
    '- "U is absent"\n\n'
    "For each claim extract:\n"
    "- pattern: a short description of what was claimed\n"
    "- claimed_count: integer 0-5 (how many of the 5 sessions the agent claims it appears in)\n"
    "- quote: the exact phrase from the text supporting this claim (max 15 words)\n\n"
    'Return ONLY valid JSON. If no claims, return: {"claims": []}\n'
    'Example with one claim: {"claims": [{"pattern": "section breaks", "claimed_count": 5, "quote": "section breaks appear in every session"}]}\n\n'
    "REFLECTION TEXT:\n"
    "{reflection}"
)


VERIFY_PROMPT_TEMPLATE = (
    "You are checking whether a specific pattern appears in a session text.\n\n"
    "Pattern to check: {pattern}\n\n"
    "Does this pattern appear in the session text below? Answer with ONLY yes or no.\n\n"
    "SESSION TEXT:\n{session_text}"
)


def call_claude(prompt: str, timeout: int = 45) -> str | None:
    try:
        result = subprocess.run(
            ["claude", "--print", prompt],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def extract_claims(reflection: str) -> list[dict]:
    prompt = EXTRACT_PROMPT_TEMPLATE.replace("{reflection}", reflection[:3000])
    response = call_claude(prompt, timeout=60)
    if not response:
        return []
    # Find JSON in response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return []
    try:
        data = json.loads(json_match.group())
        return data.get("claims", [])
    except json.JSONDecodeError:
        return []


def verify_claim(pattern: str, session_text: str) -> bool:
    prompt = (VERIFY_PROMPT_TEMPLATE
              .replace("{pattern}", pattern)
              .replace("{session_text}", session_text[:1500]))
    response = call_claude(prompt, timeout=30)
    if not response:
        return False
    return response.lower().strip().startswith("yes")


def load_existing() -> dict:
    if RESULTS_OUT.exists():
        try:
            return json.loads(RESULTS_OUT.read_text())
        except Exception:
            pass
    return {"evolutions": []}


def save_results(data: dict):
    with open(str(RESULTS_OUT), "w") as f:
        json.dump(data, f, indent=2)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SELF-REFLECTION DIAGNOSIS ACCURACY ANALYSIS")
    print("=" * 70)
    print()

    existing = load_existing()
    processed_paths = {e["evolution_path"] for e in existing["evolutions"]}
    print(f"Previously processed: {len(processed_paths)} evolutions")

    all_evolutions = []
    for instance_key, paths in INSTANCES.items():
        evo_dir = paths["evolutions"]
        if not evo_dir.exists():
            continue
        evo_files = sorted(f for f in evo_dir.glob("*.md")
                           if "_raw" not in f.name)
        print(f"  {instance_key}: {len(evo_files)} evolutions")
        for f in evo_files:
            all_evolutions.append((instance_key, paths["sessions"], f))

    print(f"\nTotal evolutions to process: {len(all_evolutions)}")
    print(f"Remaining: {len(all_evolutions) - len(processed_paths)}")
    print()

    results = list(existing["evolutions"])

    for i, (instance_key, session_dir, evo_path) in enumerate(all_evolutions):
        if str(evo_path) in processed_paths:
            print(f"  [{i+1:03d}/{len(all_evolutions)}] SKIP {evo_path.name}")
            continue

        print(f"  [{i+1:03d}/{len(all_evolutions)}] {instance_key} / {evo_path.name}")

        evo = load_evolution(evo_path)
        if not evo or not evo["session_files"]:
            print(f"    → could not parse, skipping")
            results.append({
                "evolution_path": str(evo_path),
                "instance": instance_key,
                "status": "parse_error",
            })
            save_results({"evolutions": results})
            continue

        n_sessions = len(evo["session_files"])
        print(f"    sessions: {n_sessions}  |  extracting claims...", end=" ", flush=True)

        # Extract claims from reflection
        claims = extract_claims(evo["reflection"])
        print(f"{len(claims)} claims found")

        if not claims:
            results.append({
                "evolution_path": str(evo_path),
                "instance": instance_key,
                "status": "no_claims",
                "n_sessions": n_sessions,
                "claims": [],
            })
            save_results({"evolutions": results})
            continue

        # Load session texts
        session_texts = {}
        for fname in evo["session_files"]:
            text = load_session_text(session_dir, fname)
            if text:
                session_texts[fname] = text

        # Verify each claim against each session
        claim_results = []
        for claim in claims:
            pattern = claim["pattern"]
            claimed_count = claim.get("claimed_count", -1)

            if claimed_count < 0 or claimed_count > 5:
                continue

            actual_count = 0
            session_results = []
            for fname, text in session_texts.items():
                found = verify_claim(pattern, text)
                session_results.append({"session": fname, "found": found})
                if found:
                    actual_count += 1

            # Accuracy: is the claimed count correct?
            # Allow ±1 tolerance for borderline cases
            exact_match = (actual_count == claimed_count)
            close_match = abs(actual_count - claimed_count) <= 1
            direction = ("correct" if exact_match
                        else "over" if claimed_count > actual_count
                        else "under")

            claim_results.append({
                "pattern": pattern,
                "quote": claim.get("quote", ""),
                "claimed_count": claimed_count,
                "actual_count": actual_count,
                "n_sessions_checked": len(session_texts),
                "exact_match": exact_match,
                "close_match": close_match,
                "direction": direction,
                "sessions": session_results,
            })

            status_icon = "✓" if exact_match else ("~" if close_match else "✗")
            print(f"    {status_icon} '{pattern}': claimed {claimed_count}/5, "
                  f"actual {actual_count}/5  [{direction}]")

        results.append({
            "evolution_path": str(evo_path),
            "instance": instance_key,
            "status": "ok",
            "n_sessions": n_sessions,
            "n_claims": len(claim_results),
            "claims": claim_results,
        })
        save_results({"evolutions": results})

    # ── Aggregate statistics ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    all_claims = []
    for r in results:
        if r.get("status") == "ok":
            all_claims.extend(r.get("claims", []))

    if not all_claims:
        print("No claims to analyse.")
        return

    n_total    = len(all_claims)
    n_exact    = sum(1 for c in all_claims if c["exact_match"])
    n_close    = sum(1 for c in all_claims if c["close_match"])
    n_over     = sum(1 for c in all_claims if c["direction"] == "over")
    n_under    = sum(1 for c in all_claims if c["direction"] == "under")

    print(f"Total frequency claims analysed: {n_total}")
    print(f"Exact accuracy (claimed = actual):     {n_exact}/{n_total} = {100*n_exact/n_total:.1f}%")
    print(f"Close accuracy (within ±1 session):   {n_close}/{n_total} = {100*n_close/n_total:.1f}%")
    print(f"Over-claims (claimed > actual):        {n_over}/{n_total} = {100*n_over/n_total:.1f}%")
    print(f"Under-claims (claimed < actual):       {n_under}/{n_total} = {100*n_under/n_total:.1f}%")
    print()

    # By instance
    print("── Per instance ──────────────────────────────────────────────────")
    for instance_key in INSTANCES:
        inst_claims = [c for r in results
                       if r.get("instance") == instance_key
                       for c in r.get("claims", [])]
        if not inst_claims:
            continue
        exact = sum(1 for c in inst_claims if c["exact_match"])
        print(f"  {instance_key:10s}: {exact}/{len(inst_claims)} exact "
              f"({100*exact/len(inst_claims):.0f}%)")

    print()

    # Most commonly over/under-diagnosed patterns
    from collections import Counter
    over_patterns  = Counter(c["pattern"] for c in all_claims if c["direction"] == "over")
    under_patterns = Counter(c["pattern"] for c in all_claims if c["direction"] == "under")

    if over_patterns:
        print("── Most over-claimed patterns (agent exaggerates frequency) ──────")
        for pat, count in over_patterns.most_common(5):
            print(f"  {count}x  {pat}")
    if under_patterns:
        print()
        print("── Most under-claimed patterns (agent underestimates frequency) ──")
        for pat, count in under_patterns.most_common(5):
            print(f"  {count}x  {pat}")

    print()

    # Save final summary to results
    summary = {
        "n_total_claims": n_total,
        "exact_accuracy": round(100 * n_exact / n_total, 1),
        "close_accuracy": round(100 * n_close / n_total, 1),
        "over_claim_rate": round(100 * n_over / n_total, 1),
        "under_claim_rate": round(100 * n_under / n_total, 1),
    }
    final = {"summary": summary, "evolutions": results}
    save_results(final)
    print(f"Results saved to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()
