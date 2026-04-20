#!/usr/bin/env python3
"""
Generate three publication-quality figures for the DMN project.
"""

import json
import re
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine, cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DIR = Path(__file__).parent
OUTPUT_DIR = DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

BG = "#1a1a2e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
MUTED_TEXT = "#888888"

COLOURS = {
    "alpha": "#e74c3c",
    "beta": "#3498db",
    "gamma": "#2ecc71",
    "null": "#95a5a6",
    "perturb": "#9b59b6",
}


def style_ax(ax, fig):
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, color=GRID_COLOR, alpha=0.3, linewidth=0.5)


# ── Load sessions ────────────────────────────────────────────────────────────
def load_sessions(sessions_dir):
    sessions = []
    if not sessions_dir.exists():
        return sessions
    for f in sorted(sessions_dir.glob("*.md")):
        text = f.read_text()
        lines = text.split("\n")
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip() == "---":
                body_start = i + 1
                break
        body = "\n".join(lines[body_start:]).strip()
        if not body:
            continue
        num_match = re.search(r"^#\s*(\d+)", text)
        num = int(num_match.group(1)) if num_match else 0
        sessions.append({"num": num, "body": body})
    return sessions


# ── FIGURE 1: Perturbation Recovery ──────────────────────────────────────────
def figure_perturbation_recovery():
    print("Figure 1: Perturbation Recovery")

    # Load perturb sessions
    sessions = load_sessions(DIR / "instances" / "perturb" / "sessions")
    sessions.sort(key=lambda s: s["num"])
    print(f"  Loaded {len(sessions)} perturb sessions")

    # Embed
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [s["body"] for s in sessions]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)

    session_nums = [s["num"] for s in sessions]

    # Split into perturbed and non-perturbed
    non_perturbed_mask = [i for i, s in enumerate(sessions) if s["num"] % 4 != 0]
    non_perturbed_embs = embeddings[non_perturbed_mask]

    # Centroid of non-perturbed
    centroid = np.mean(non_perturbed_embs, axis=0)

    # Cosine distance from centroid for every session
    distances = []
    for emb in embeddings:
        distances.append(cosine(emb, centroid))
    distances = np.array(distances)

    # Baseline stats (non-perturbed sessions)
    baseline_dists = distances[non_perturbed_mask]
    baseline_mean = np.mean(baseline_dists)
    baseline_std = np.std(baseline_dists)

    # Perturbation session numbers
    perturb_nums = sorted(set(s["num"] for s in sessions if s["num"] % 4 == 0))

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    style_ax(ax, fig)

    # Baseline band
    ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
               color="#3498db", alpha=0.12, zorder=1)
    ax.axhline(baseline_mean, color="#3498db", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Baseline mean ({baseline_mean:.3f})", zorder=2)

    # Main line
    ax.plot(session_nums, distances, color="#9b59b6", linewidth=1.8, alpha=0.9,
            zorder=4, label="Perturb instance")
    ax.scatter(session_nums, distances, color="#9b59b6", s=18, alpha=0.7, zorder=5)

    # Perturbation events
    first = True
    for pn in perturb_nums:
        ax.axvline(pn, color="#e74c3c", linestyle="--", linewidth=0.9, alpha=0.55,
                   zorder=3, label="Perturbation event" if first else None)
        first = False

    ax.set_xlabel("Session Number", color=MUTED_TEXT, fontsize=12)
    ax.set_ylabel("Cosine Distance from Centroid", color=MUTED_TEXT, fontsize=12)
    ax.set_title("Perturbation Recovery — Distance from Attractor Centroid",
                 color=TEXT_COLOR, fontsize=15, pad=15, fontweight="bold")
    ax.legend(loc="upper right", facecolor=BG, edgecolor="#444",
              labelcolor=TEXT_COLOR, fontsize=10, framealpha=0.9)
    ax.set_xlim(0, max(session_nums) + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

    plt.tight_layout()
    out = OUTPUT_DIR / "perturbation_recovery.png"
    plt.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── FIGURE 2: Null Comparison ────────────────────────────────────────────────
def figure_null_comparison():
    print("\nFigure 2: Null Comparison")

    # Load embeddings from npz (faster than re-embedding)
    data = np.load(OUTPUT_DIR / "embeddings.npz")
    instances = {}
    for name in ["null", "alpha", "beta", "gamma"]:
        instances[name] = data[name]
        print(f"  {name}: {instances[name].shape[0]} embeddings")

    # Compute within-instance pairwise cosine distances
    within_stats = {}
    for name, embs in instances.items():
        dists = cdist(embs, embs, metric="cosine")
        # Upper triangle only (no self-pairs)
        triu = dists[np.triu_indices_from(dists, k=1)]
        within_stats[name] = {"mean": np.mean(triu), "std": np.std(triu)}
        print(f"  Within-{name}: mean={within_stats[name]['mean']:.4f}, std={within_stats[name]['std']:.4f}")

    # Null to all controls cross-instance
    controls = ["alpha", "beta", "gamma"]
    null_to_ctrl_dists = []
    for ctrl in controls:
        cross = cdist(instances["null"], instances[ctrl], metric="cosine")
        null_to_ctrl_dists.extend(cross.flatten().tolist())
    null_to_ctrl = {"mean": np.mean(null_to_ctrl_dists), "std": np.std(null_to_ctrl_dists)}
    print(f"  Null->Controls: mean={null_to_ctrl['mean']:.4f}")

    # Control to control cross-instance
    ctrl_pairs = [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")]
    ctrl_to_ctrl_dists = []
    for a, b in ctrl_pairs:
        cross = cdist(instances[a], instances[b], metric="cosine")
        ctrl_to_ctrl_dists.extend(cross.flatten().tolist())
    ctrl_to_ctrl = {"mean": np.mean(ctrl_to_ctrl_dists), "std": np.std(ctrl_to_ctrl_dists)}
    print(f"  Ctrl->Ctrl: mean={ctrl_to_ctrl['mean']:.4f}")

    # Plot grouped bars
    groups = ["Within-null", "Within-alpha", "Within-beta", "Within-gamma",
              "Null→Controls", "Ctrl→Ctrl"]
    means = [
        within_stats["null"]["mean"],
        within_stats["alpha"]["mean"],
        within_stats["beta"]["mean"],
        within_stats["gamma"]["mean"],
        null_to_ctrl["mean"],
        ctrl_to_ctrl["mean"],
    ]
    stds = [
        within_stats["null"]["std"],
        within_stats["alpha"]["std"],
        within_stats["beta"]["std"],
        within_stats["gamma"]["std"],
        null_to_ctrl["std"],
        ctrl_to_ctrl["std"],
    ]
    bar_colors = [
        COLOURS["null"],
        COLOURS["alpha"],
        COLOURS["beta"],
        COLOURS["gamma"],
        "#cccccc",
        "#f0f0f0",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    style_ax(ax, fig)
    ax.grid(axis="x", visible=False)

    x = np.arange(len(groups))
    bars = ax.bar(x, means, yerr=stds, width=0.6, color=bar_colors, edgecolor="#444",
                  linewidth=0.8, capsize=5, error_kw={"elinewidth": 1.2, "capthick": 1.2,
                                                       "ecolor": "#aaa"}, zorder=4)

    # Value labels on bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{m:.3f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=10,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Mean Cosine Distance", color=MUTED_TEXT, fontsize=12)
    ax.set_title("Null Baseline vs Control Instances — Distance Comparison",
                 color=TEXT_COLOR, fontsize=15, pad=15, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25)

    plt.tight_layout()
    out = OUTPUT_DIR / "null_comparison.png"
    plt.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── FIGURE 3: Phase Heatmap ─────────────────────────────────────────────────
def figure_phase_heatmap():
    print("\nFigure 3: Phase Heatmap")

    # Load convergence data from results.json
    with open(OUTPUT_DIR / "results.json") as f:
        results = json.load(f)

    convergence = results["convergence"]

    # We need alpha-beta, alpha-gamma, beta-gamma
    pair_keys = {
        "alpha–beta": "alpha–beta",
        "alpha–gamma": "alpha–gamma",
        "beta–gamma": "beta–gamma",
    }

    # But convergence stores cosine *similarity* — we need distance
    # Actually, looking at the data: values ~0.5-0.7 — these look like similarities
    # from compute_convergence which does 1 - cosine(a,b) = similarity
    # So distance = 1 - similarity

    # Phase windows: 1-20, 21-40, 41-60, 61-80, 81-100
    # Convergence midpoints go from 2 to ~97
    phases = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    phase_labels = ["1–20", "21–40", "41–60", "61–80", "81–100"]
    pair_labels = ["alpha–beta", "alpha–gamma", "beta–gamma"]

    heatmap_data = np.zeros((len(pair_labels), len(phases)))

    for row, pair_name in enumerate(pair_labels):
        sims = convergence.get(pair_name, [])
        for col, (lo, hi) in enumerate(phases):
            # Filter points whose midpoint falls in this phase
            phase_sims = [s for midpt, s in sims if lo <= midpt < hi]
            if phase_sims:
                # Convert similarity to distance
                heatmap_data[row, col] = 1 - np.mean(phase_sims)
            else:
                heatmap_data[row, col] = np.nan

    print(f"  Heatmap data:\n{heatmap_data}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    style_ax(ax, fig)
    ax.grid(False)

    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=np.nanmin(heatmap_data) * 0.95,
                   vmax=np.nanmax(heatmap_data) * 1.05)

    # Annotate cells
    for i in range(len(pair_labels)):
        for j in range(len(phases)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmean(heatmap_data) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phase_labels, color=TEXT_COLOR, fontsize=11)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, color=TEXT_COLOR, fontsize=11)
    ax.set_xlabel("Phase (Session Window)", color=MUTED_TEXT, fontsize=12)
    ax.set_ylabel("Instance Pair", color=MUTED_TEXT, fontsize=12)
    ax.set_title("Cross-Instance Convergence by Phase",
                 color=TEXT_COLOR, fontsize=15, pad=15, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label("Mean Cosine Distance", color=MUTED_TEXT, fontsize=11)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    plt.tight_layout()
    out = OUTPUT_DIR / "phase_heatmap.png"
    plt.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    figure_perturbation_recovery()
    figure_null_comparison()
    figure_phase_heatmap()
    print("\nDone — all three figures saved.")
