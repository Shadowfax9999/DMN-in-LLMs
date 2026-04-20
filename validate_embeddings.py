"""
Embedding Validation
====================
Samples 50 session pairs stratified across cosine distance quantiles,
asks Claude to rate thematic similarity, and correlates the ratings
with cosine distance to validate the embedding methodology.

Uses the existing embeddings.npz and rates pairs via claude CLI.
"""

import json
import os
import random
import subprocess
import sys
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path("/Users/charliemurray/Documents/creativity work")
EMBEDDINGS = BASE / "analysis" / "embeddings.npz"
RESULTS_OUT = BASE / "embedding_validation.json"
N_PAIRS = 50
RANDOM_SEED = 42

# Rating prompt sent to Claude for each pair
RATING_PROMPT = """You are rating the thematic similarity between two pieces of AI-generated stream-of-consciousness writing.

Rate their THEMATIC similarity (not stylistic or linguistic) on this scale:
0 = completely different topics/themes
1 = slightly related (share a vague domain but little else)
2 = moderately similar (overlap in subject matter)
3 = quite similar (clearly about the same kinds of things)
4 = almost identical in theme (same topic, similar observations)

Reply with ONLY a single integer 0-4. No explanation.

--- SESSION A ---
{text_a}

--- SESSION B ---
{text_b}

Rating (0-4):"""


def load_sessions_for_instance(instance_key: str) -> list[tuple[str, str]]:
    """Return list of (path, text) for a given instance key."""
    if instance_key == "main":
        session_dir = BASE / "sessions"
    else:
        session_dir = BASE / "instances" / instance_key / "sessions"

    if not session_dir.exists():
        return []

    files = sorted(session_dir.glob("*.md"))
    sessions = []
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="replace")
        # Strip YAML frontmatter
        lines = raw.splitlines()
        dashes = [i for i, l in enumerate(lines) if l.strip() == "---"]
        if len(dashes) >= 2:
            lines = lines[:dashes[0]] + lines[dashes[1] + 1:]
        text = "\n".join(lines).strip()
        if text:
            sessions.append((str(f), text))
    return sessions


def truncate(text: str, max_chars: int = 1200) -> str:
    """Truncate text to max_chars, keeping start."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a, b))


def rate_pair_with_claude(text_a: str, text_b: str) -> int | None:
    """Call claude CLI to rate the thematic similarity of a pair."""
    prompt = RATING_PROMPT.format(
        text_a=truncate(text_a, 1200),
        text_b=truncate(text_b, 1200),
    )
    try:
        result = subprocess.run(
            ["claude", "--print", prompt],
            capture_output=True, text=True, timeout=60
        )
        out = result.stdout.strip()
        # Extract the last integer in the output
        for token in reversed(out.split()):
            try:
                rating = int(token)
                if 0 <= rating <= 4:
                    return rating
            except ValueError:
                continue
        print(f"  [WARN] Could not parse rating from: {out!r}")
        return None
    except subprocess.TimeoutExpired:
        print("  [WARN] Claude call timed out")
        return None
    except Exception as e:
        print(f"  [WARN] Claude call failed: {e}")
        return None


def main():
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("EMBEDDING VALIDATION")
    print("=" * 70)
    print()

    # Load embeddings
    data = np.load(str(EMBEDDINGS), allow_pickle=True)

    # Build flat index: list of (instance_key, session_idx, embedding)
    print("Loading embeddings and session texts...")
    all_entries = []
    for key in data.keys():
        embs = data[key]  # shape (N, 384)
        sessions = load_sessions_for_instance(key)
        n = min(len(embs), len(sessions))
        for i in range(n):
            all_entries.append({
                "instance": key,
                "idx": i,
                "embedding": embs[i],
                "path": sessions[i][0],
                "text": sessions[i][1],
            })

    print(f"  Total sessions available: {len(all_entries)}")
    print()

    # Compute all pairwise cosine distances for a random sample to get distribution
    print("Computing distance distribution for stratified sampling...")
    sample_size = min(300, len(all_entries))
    sampled = rng.sample(all_entries, sample_size)
    candidate_pairs = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            d = cosine_distance(sampled[i]["embedding"], sampled[j]["embedding"])
            candidate_pairs.append((d, sampled[i], sampled[j]))

    candidate_pairs.sort(key=lambda x: x[0])
    print(f"  Candidate pairs: {len(candidate_pairs)}")
    print(f"  Distance range: {candidate_pairs[0][0]:.4f} – {candidate_pairs[-1][0]:.4f}")
    print()

    # Stratified sample: 10 pairs per quintile
    n_quintiles = 5
    pairs_per_quintile = N_PAIRS // n_quintiles
    quintile_size = len(candidate_pairs) // n_quintiles
    selected_pairs = []
    for q in range(n_quintiles):
        start = q * quintile_size
        end = start + quintile_size
        chunk = candidate_pairs[start:end]
        chosen = rng.sample(chunk, min(pairs_per_quintile, len(chunk)))
        selected_pairs.extend(chosen)
        d_vals = [p[0] for p in chosen]
        print(f"  Quintile {q+1}: {len(chosen)} pairs, distance range "
              f"{min(d_vals):.3f}–{max(d_vals):.3f}")

    print(f"\nSelected {len(selected_pairs)} pairs total")
    print()

    # Load any existing partial results so we can resume
    existing = {}
    if RESULTS_OUT.exists():
        try:
            prev = json.loads(RESULTS_OUT.read_text())
            for r in prev.get("pairs", []):
                if r.get("claude_rating") is not None:
                    existing[r["pair_idx"]] = r
            print(f"Resuming — {len(existing)} pairs already rated")
        except Exception:
            pass

    # Rate each pair
    print("Rating pairs with Claude...")
    print("-" * 70)
    results = []
    for i, (dist, entry_a, entry_b) in enumerate(selected_pairs):
        if i in existing:
            print(f"  [{i+1:02d}/{len(selected_pairs)}] SKIP (already rated={existing[i]['claude_rating']})")
            results.append(existing[i])
            continue

        print(f"  [{i+1:02d}/{len(selected_pairs)}] "
              f"{entry_a['instance']}[{entry_a['idx']}] vs "
              f"{entry_b['instance']}[{entry_b['idx']}]  "
              f"dist={dist:.4f}", end="  ", flush=True)

        rating = rate_pair_with_claude(entry_a["text"], entry_b["text"])
        print(f"rating={rating}")

        result = {
            "pair_idx": i,
            "instance_a": entry_a["instance"],
            "session_idx_a": entry_a["idx"],
            "path_a": entry_a["path"],
            "instance_b": entry_b["instance"],
            "session_idx_b": entry_b["idx"],
            "path_b": entry_b["path"],
            "cosine_distance": float(dist),
            "claude_rating": rating,
        }
        results.append(result)

        # Save incrementally after each pair
        partial_output = {
            "n_pairs": len(selected_pairs),
            "n_rated": sum(1 for r in results if r.get("claude_rating") is not None),
            "pairs": results,
        }
        with open(str(RESULTS_OUT), "w") as f:
            json.dump(partial_output, f, indent=2)

    # Filter to rated pairs
    rated = [r for r in results if r["claude_rating"] is not None]
    print()
    print(f"Rated pairs: {len(rated)}/{len(results)}")

    if len(rated) < 10:
        print("Too few rated pairs for meaningful correlation. Exiting.")
        return

    distances = np.array([r["cosine_distance"] for r in rated])
    ratings   = np.array([r["claude_rating"]    for r in rated])

    # Pearson correlation (cosine distance vs similarity rating — expect negative)
    pearson_r, pearson_p = stats.pearsonr(distances, ratings)
    # Spearman (rank-based, more robust)
    spearman_r, spearman_p = stats.spearmanr(distances, ratings)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Pairs rated: {len(rated)}")
    print(f"Rating distribution:")
    for rating in range(5):
        count = int(np.sum(ratings == rating))
        bar = "█" * count
        print(f"  {rating}: {bar} ({count})")
    print()
    print(f"Cosine distance vs Claude similarity rating:")
    print(f"  Pearson r  = {pearson_r:+.4f}  (p = {pearson_p:.4f})")
    print(f"  Spearman ρ = {spearman_r:+.4f}  (p = {spearman_p:.4f})")
    print()

    if abs(spearman_r) > 0.4 and spearman_p < 0.05:
        print("✓ VALIDATED: Cosine distance correlates significantly with")
        print("  Claude's thematic similarity judgments.")
        verdict = "validated"
    elif abs(spearman_r) > 0.2:
        print("~ PARTIAL: Moderate correlation — embeddings capture some but")
        print("  not all thematic variation relevant to this use case.")
        verdict = "partial"
    else:
        print("✗ WEAK: Low correlation — embeddings may be capturing style/prose")
        print("  rather than thematic content.")
        verdict = "weak"

    print()
    print(f"Interpretation: Expected direction is negative (higher distance → lower")
    print(f"similarity rating). Pearson r = {pearson_r:+.3f} {'✓' if pearson_r < 0 else '✗ (unexpected sign)'}")

    # Save
    output = {
        "n_pairs": len(selected_pairs),
        "n_rated": len(rated),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "verdict": verdict,
        "pairs": results,
    }
    with open(str(RESULTS_OUT), "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Full results saved to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()
