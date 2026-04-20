#!/usr/bin/env python3
"""
DMN Attractor Analysis
Computes session embeddings, cross-instance convergence, and UMAP visualisation.
"""

import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
DIR = Path(__file__).parent
INSTANCES = {
    "main": DIR / "sessions",
    "alpha": DIR / "instances" / "alpha" / "sessions",
    "beta": DIR / "instances" / "beta" / "sessions",
    "gamma": DIR / "instances" / "gamma" / "sessions",
    "replay": DIR / "instances" / "replay" / "sessions",
    "perturb": DIR / "instances" / "perturb" / "sessions",
    "switch": DIR / "instances" / "switch" / "sessions",
    "null": DIR / "instances" / "null" / "sessions",
    "sonnet": DIR / "instances" / "sonnet" / "sessions",
    "sonnet-null": DIR / "instances" / "sonnet-null" / "sessions",
    "sonnet-alpha": DIR / "instances" / "sonnet-alpha" / "sessions",
    "sonnet-beta": DIR / "instances" / "sonnet-beta" / "sessions",
    "sonnet-gamma": DIR / "instances" / "sonnet-gamma" / "sessions",
    "sonnet-perturb": DIR / "instances" / "sonnet-perturb" / "sessions",
    "llama-null": DIR / "instances" / "llama-null" / "sessions",
    "llama-alpha": DIR / "instances" / "llama-alpha" / "sessions",
    "llama-beta": DIR / "instances" / "llama-beta" / "sessions",
    "llama-gamma": DIR / "instances" / "llama-gamma" / "sessions",
    "llama-perturb": DIR / "instances" / "llama-perturb" / "sessions",
    "gpt-null": DIR / "instances" / "gpt-null" / "sessions",
    "gpt-alpha": DIR / "instances" / "gpt-alpha" / "sessions",
    "gpt-beta": DIR / "instances" / "gpt-beta" / "sessions",
    "gpt-gamma": DIR / "instances" / "gpt-gamma" / "sessions",
    "gpt-perturb": DIR / "instances" / "gpt-perturb" / "sessions",
    "perturb-no-drift": DIR / "instances" / "perturb-no-drift" / "sessions",
}
OUTPUT_DIR = DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Load sessions ─────────────────────────────────────────────────────────────
def load_sessions(sessions_dir):
    """Load all session files, return list of dicts sorted by filename."""
    sessions = []
    if not sessions_dir.exists():
        return sessions
    for f in sorted(sessions_dir.glob("*.md")):
        text = f.read_text()
        # Strip the header (# N, seed line, ---)
        lines = text.split("\n")
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip() == "---":
                body_start = i + 1
                break
        body = "\n".join(lines[body_start:]).strip()
        if not body:
            continue

        # Extract session number
        num_match = re.search(r"^#\s*(\d+)", text)
        num = int(num_match.group(1)) if num_match else 0

        # Extract seed concept
        concept = None
        seed_match = re.search(r"\*seed:.*?·\s*(\w[\w\s]*)\*", text)
        if seed_match:
            concept = seed_match.group(1).strip()

        sessions.append({
            "file": f.name,
            "num": num,
            "body": body,
            "concept": concept,
            "word_count": len(body.split()),
        })
    return sessions


# ── Compute embeddings ────────────────────────────────────────────────────────
def compute_embeddings(sessions, model):
    """Compute embeddings for all sessions using sentence-transformers."""
    texts = [s["body"] for s in sessions]
    print(f"  Embedding {len(texts)} sessions...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
    return embeddings


# ── Cross-instance convergence ────────────────────────────────────────────────
def compute_convergence(embeddings_by_instance, window=5):
    """
    Compute pairwise cosine similarity between instances over sliding windows.
    Returns a dict of {pair_name: [(window_midpoint, similarity), ...]}.
    """
    from scipy.spatial.distance import cosine

    instance_names = list(embeddings_by_instance.keys())
    pairs = []
    for i in range(len(instance_names)):
        for j in range(i + 1, len(instance_names)):
            pairs.append((instance_names[i], instance_names[j]))

    convergence = {}
    for name_a, name_b in pairs:
        emb_a = embeddings_by_instance[name_a]
        emb_b = embeddings_by_instance[name_b]
        min_len = min(len(emb_a), len(emb_b))
        if min_len < window:
            continue

        pair_name = f"{name_a}–{name_b}"
        similarities = []
        for start in range(0, min_len - window + 1):
            end = start + window
            # Average embedding for each instance in this window
            avg_a = np.mean(emb_a[start:end], axis=0)
            avg_b = np.mean(emb_b[start:end], axis=0)
            sim = 1 - cosine(avg_a, avg_b)
            midpoint = start + window // 2
            similarities.append((midpoint, sim))
        convergence[pair_name] = similarities

    return convergence


# ── Lyapunov-style convergence rate ───────────────────────────────────────────
def compute_convergence_rate(embeddings_by_instance):
    """
    Compute the average pairwise distance between instances at each session index.
    A decreasing trend = convergent dynamics (negative Lyapunov-like exponent).
    """
    from scipy.spatial.distance import cosine

    instance_names = [n for n in embeddings_by_instance if n != "main"]
    if len(instance_names) < 2:
        return []

    min_len = min(len(embeddings_by_instance[n]) for n in instance_names)
    distances_over_time = []

    for t in range(min_len):
        dists = []
        for i in range(len(instance_names)):
            for j in range(i + 1, len(instance_names)):
                d = cosine(
                    embeddings_by_instance[instance_names[i]][t],
                    embeddings_by_instance[instance_names[j]][t],
                )
                dists.append(d)
        distances_over_time.append((t, np.mean(dists)))

    return distances_over_time


# ── UMAP projection ──────────────────────────────────────────────────────────
def compute_umap(embeddings_by_instance):
    """Project all session embeddings to 2D using UMAP."""
    import umap

    all_embeddings = []
    labels = []
    indices = []
    for name, embs in embeddings_by_instance.items():
        for i, emb in enumerate(embs):
            all_embeddings.append(emb)
            labels.append(name)
            indices.append(i)

    all_embeddings = np.array(all_embeddings)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    projection = reducer.fit_transform(all_embeddings)

    return projection, labels, indices


# ── Plot everything ───────────────────────────────────────────────────────────
def plot_all(convergence, distances, projection, labels, indices, embeddings_by_instance):
    """Generate all analysis plots."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    colours = {
        "main": "#888888",
        "alpha": "#e74c3c",
        "beta": "#3498db",
        "gamma": "#2ecc71",
        "replay": "#f39c12",
        "perturb": "#9b59b6",
        "switch": "#1abc9c",
        "null": "#95a5a6",
        "sonnet": "#e67e22",
        "sonnet-null": "#bdc3c7",
        "sonnet-alpha": "#d35400",
        "sonnet-beta": "#2980b9",
        "sonnet-gamma": "#27ae60",
        "sonnet-perturb": "#8e44ad",
        "llama-null": "#c0392b",
        "llama-alpha": "#e74c3c",
        "llama-beta": "#e55039",
        "llama-gamma": "#cb4335",
        "llama-perturb": "#922b21",
        "gpt-null": "#ff6b6b",
        "gpt-alpha": "#ee5a24",
        "gpt-beta": "#0984e3",
        "gpt-gamma": "#00b894",
        "gpt-perturb": "#6c5ce7",
        "perturb-no-drift": "#fd79a8",
    }

    # ── Figure 1: UMAP attractor map ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.set_facecolor("#1a1a2e")

    # Get all unique instance names from labels
    all_names = list(dict.fromkeys(labels))  # preserves order, removes dupes

    for name in all_names:
        mask = [i for i, l in enumerate(labels) if l == name]
        if not mask:
            continue
        x = projection[mask, 0]
        y = projection[mask, 1]
        colour = colours.get(name, "#888888")

        # Trajectory lines — subtle, gives it the strange attractor look
        if len(x) > 1:
            ax.plot(x, y, color=colour, alpha=0.35, linewidth=0.8, zorder=2)

        # Fade from light to dark as sessions progress
        alphas = np.linspace(0.4, 1.0, len(x))
        for xi, yi, ai in zip(x, y, alphas):
            ax.scatter(xi, yi, color=colour, alpha=ai, s=25, zorder=5)
        ax.scatter([], [], color=colour, s=25, label=name, alpha=0.8)  # legend entry

        # Mark first session with a ring
        ax.scatter(x[0], y[0], color=colour, s=120, marker="o",
                   edgecolors="white", linewidth=1.5, zorder=10, alpha=0.9)

    ax.set_title("DMN Attractor Map — Session Positions in Embedding Space",
                 color="white", fontsize=14, pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="white", fontsize=10, markerscale=1.5)
    ax.set_xlabel("UMAP-1", color="#666")
    ax.set_ylabel("UMAP-2", color="#666")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umap_attractor_map.png", dpi=150, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'umap_attractor_map.png'}")

    # ── Figure 2: Convergence curves ─────────────────────────────────────────
    if convergence:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.set_facecolor("#1a1a2e")
        fig.set_facecolor("#1a1a2e")

        pair_colours = {
            "alpha–beta": "#e67e22",
            "alpha–gamma": "#9b59b6",
            "beta–gamma": "#1abc9c",
        }
        for pair_name, sims in convergence.items():
            if not sims:
                continue
            x = [s[0] for s in sims]
            y = [s[1] for s in sims]
            colour = pair_colours.get(pair_name, "#ffffff")
            ax.plot(x, y, color=colour, linewidth=2, label=pair_name)
            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), color=colour, linewidth=1, linestyle="--", alpha=0.5)

        ax.set_title("Cross-Instance Convergence (Cosine Similarity, 5-session windows)",
                     color="white", fontsize=13, pad=15)
        ax.set_xlabel("Session index", color="#666")
        ax.set_ylabel("Cosine similarity", color="#666")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "convergence_curves.png", dpi=150, facecolor="#1a1a2e")
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'convergence_curves.png'}")

    # ── Figure 3: Distance over time (Lyapunov-style) ────────────────────────
    if distances:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.set_facecolor("#1a1a2e")
        fig.set_facecolor("#1a1a2e")

        x = [d[0] for d in distances]
        y = [d[1] for d in distances]
        ax.plot(x, y, color="#e74c3c", linewidth=2, label="Mean pairwise distance")

        # Trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), color="#e74c3c", linewidth=1, linestyle="--", alpha=0.5)

        direction = "CONVERGING ↓" if z[0] < 0 else "DIVERGING ↑"
        ax.text(0.02, 0.95, f"Trend: {direction} (slope: {z[0]:.6f})",
                transform=ax.transAxes, color="white", fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", alpha=0.8))

        ax.set_title("Inter-Instance Distance Over Time (lower = more similar)",
                     color="white", fontsize=13, pad=15)
        ax.set_xlabel("Session index", color="#666")
        ax.set_ylabel("Mean cosine distance", color="#666")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "distance_over_time.png", dpi=150, facecolor="#1a1a2e")
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'distance_over_time.png'}")

    # ── Figure 4: Per-instance entropy over time ──────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_facecolor("#1a1a2e")
    fig.set_facecolor("#1a1a2e")

    for name, embs in embeddings_by_instance.items():
        if len(embs) < 5:
            continue
        # Compute rolling vocabulary entropy using session word distributions
        # (approximated by embedding variance within a sliding window)
        window = 5
        variances = []
        for i in range(len(embs) - window + 1):
            chunk = np.array(embs[i:i + window])
            var = np.mean(np.var(chunk, axis=0))
            variances.append((i + window // 2, var))

        x = [v[0] for v in variances]
        y = [v[1] for v in variances]
        ax.plot(x, y, color=colours[name], linewidth=1.5, label=name, alpha=0.8)

    ax.set_title("Embedding Variance Over Time (higher = more diverse sessions)",
                 color="white", fontsize=13, pad=15)
    ax.set_xlabel("Session index", color="#666")
    ax.set_ylabel("Mean embedding variance (5-session window)", color="#666")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entropy_over_time.png", dpi=150, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'entropy_over_time.png'}")


# ── Save numerical results ────────────────────────────────────────────────────
def save_results(convergence, distances, embeddings_by_instance, projection=None, labels=None):
    """Save analysis results as JSON for the dashboard."""
    results = {
        "generated": datetime.now().isoformat(),
        "session_counts": {name: len(embs) for name, embs in embeddings_by_instance.items()},
        "convergence": {
            pair: [(int(m), float(round(s, 4))) for m, s in sims]
            for pair, sims in convergence.items()
        },
        "distances": [(int(t), float(round(d, 4))) for t, d in distances],
    }

    # Compute variance over time per instance
    variance_data = {}
    window = 5
    for name, embs in embeddings_by_instance.items():
        if len(embs) < window:
            continue
        variances = []
        for i in range(len(embs) - window + 1):
            chunk = np.array(embs[i:i + window])
            var = float(np.mean(np.var(chunk, axis=0)))
            variances.append([i + window // 2, round(var, 6)])
        variance_data[name] = variances
    results["variance"] = variance_data

    # Add UMAP coordinates for interactive plotting
    if projection is not None and labels:
        umap_data = {}
        for name in dict.fromkeys(labels):
            mask = [i for i, l in enumerate(labels) if l == name]
            points = [[round(float(projection[i, 0]), 4), round(float(projection[i, 1]), 4)] for i in mask]
            umap_data[name] = points
        results["umap"] = umap_data

    # Compute summary stats
    if distances:
        early = [d for t, d in distances if t < 10]
        late = [d for t, d in distances if t >= max(0, len(distances) - 10)]
        if early and late:
            results["summary"] = {
                "early_mean_distance": float(round(np.mean(early), 4)),
                "late_mean_distance": float(round(np.mean(late), 4)),
                "convergence_ratio": float(round(np.mean(late) / np.mean(early), 4)) if np.mean(early) > 0 else None,
                "trend": "converging" if np.mean(late) < np.mean(early) else "diverging",
            }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'results.json'}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(selected=None):
    """Run analysis. If selected is a list of instance names, only analyse those."""
    print("DMN Attractor Analysis")
    print("=" * 50)

    # Filter instances if selection provided
    instances = INSTANCES
    if selected:
        instances = {k: v for k, v in INSTANCES.items() if k in selected}
        print(f"  Selected instances: {', '.join(instances.keys())}")

    # Load all sessions
    all_sessions = {}
    for name, path in instances.items():
        sessions = load_sessions(path)
        if sessions:
            all_sessions[name] = sessions
            print(f"  {name}: {len(sessions)} sessions")

    if not all_sessions:
        print("No sessions found!")
        return

    # Load embedding model
    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embeddings
    print("\nComputing embeddings...")
    embeddings_by_instance = {}
    for name, sessions in all_sessions.items():
        embs = compute_embeddings(sessions, model)
        embeddings_by_instance[name] = embs

    # Save raw embeddings for future use
    np.savez(
        OUTPUT_DIR / "embeddings.npz",
        **{name: embs for name, embs in embeddings_by_instance.items()},
    )
    print(f"  Saved: {OUTPUT_DIR / 'embeddings.npz'}")

    # Compute convergence (instances only, not main)
    print("\nComputing cross-instance convergence...")
    instance_embs = {k: v for k, v in embeddings_by_instance.items() if k != "main"}
    convergence = compute_convergence(instance_embs, window=5)

    # Compute distance over time
    print("Computing inter-instance distances...")
    distances = compute_convergence_rate(embeddings_by_instance)

    # UMAP
    print("\nComputing UMAP projection...")
    projection, labels, indices = compute_umap(embeddings_by_instance)

    # Plot
    print("\nGenerating plots...")
    plot_all(convergence, distances, projection, labels, indices, embeddings_by_instance)

    # Save results
    print("\nSaving results...")
    save_results(convergence, distances, embeddings_by_instance, projection, labels)

    # Print summary
    if distances:
        early = [d for t, d in distances if t < 10]
        late = [d for t, d in distances if t >= max(0, len(distances) - 10)]
        if early and late:
            print(f"\n{'=' * 50}")
            print(f"SUMMARY")
            print(f"  Early mean distance (sessions 0-9):  {np.mean(early):.4f}")
            print(f"  Late mean distance  (last 10):       {np.mean(late):.4f}")
            ratio = np.mean(late) / np.mean(early) if np.mean(early) > 0 else 0
            if ratio < 1:
                print(f"  Trend: CONVERGING (ratio: {ratio:.2f})")
            else:
                print(f"  Trend: DIVERGING (ratio: {ratio:.2f})")
            print(f"{'=' * 50}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("  - umap_attractor_map.png")
    print("  - convergence_curves.png")
    print("  - distance_over_time.png")
    print("  - entropy_over_time.png")
    print("  - embeddings.npz")
    print("  - results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DMN Attractor Analysis")
    parser.add_argument("instances", nargs="*", default=None,
                        help="Instance names to analyse (e.g. alpha beta gamma). "
                             "If omitted, analyses all instances.")
    args = parser.parse_args()
    main(selected=args.instances if args.instances else None)
