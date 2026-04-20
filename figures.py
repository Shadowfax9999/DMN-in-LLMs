"""
Paper-quality figures for DMN analysis.
Reads from analysis/embeddings.npz — run analyse.py first.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import norm
from pathlib import Path
from scipy.spatial.distance import cosine

ANALYSIS_DIR = Path(__file__).parent / "analysis"
FIGURES_DIR = Path(__file__).parent / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BG = "#1a1a2e"
WHITE = "#ffffff"
GREY = "#888888"

# ── Colour scheme: by model family and condition ──────────────────────────────
MODEL_COLOURS = {
    "opus":   "#e74c3c",   # red
    "sonnet": "#e67e22",   # orange
    "gpt":    "#3498db",   # blue
    "llama":  "#2ecc71",   # green
}

CONDITION_MARKERS = {
    "null":    "o",
    "evolved": "s",
    "perturb": "^",
}

def model_family(name):
    if name in ("alpha", "beta", "gamma", "null", "perturb", "perturb-no-drift", "main"):
        return "opus"
    if name.startswith("sonnet"):
        return "sonnet"
    if name.startswith("gpt"):
        return "gpt"
    if name.startswith("llama"):
        return "llama"
    return "opus"

def condition(name):
    if "null" in name:
        return "null"
    if "perturb" in name:
        return "perturb"
    if name in ("main", "alpha", "beta", "gamma",
                "sonnet-alpha", "sonnet-beta", "sonnet-gamma",
                "gpt-alpha", "gpt-beta", "gpt-gamma",
                "llama-alpha", "llama-beta", "llama-gamma"):
        return "evolved"
    return "evolved"

def cosine_dist(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def mean_within_dist(vecs):
    n = len(vecs)
    dists = [cosine_dist(vecs[i], vecs[j]) for i in range(n) for j in range(i+1, n)]
    return np.mean(dists)

# ── Load embeddings ───────────────────────────────────────────────────────────
data = np.load(ANALYSIS_DIR / "embeddings.npz", allow_pickle=True)
emb = {k: data[k] for k in data.files}

# ── Figure 1: Simplified UMAP ─────────────────────────────────────────────────
def plot_umap():
    from umap import UMAP

    # Only include the core instances for a clean plot
    core = {
        "null": emb["null"],
        "sonnet-null": emb["sonnet-null"],
        "gpt-null": emb["gpt-null"],
        "llama-null": emb["llama-null"],
        "alpha": emb["alpha"],
        "beta": emb["beta"],
        "gamma": emb["gamma"],
        "sonnet-alpha": emb["sonnet-alpha"],
        "sonnet-beta": emb["sonnet-beta"],
        "sonnet-gamma": emb["sonnet-gamma"],
        "gpt-alpha": emb["gpt-alpha"],
        "gpt-beta": emb["gpt-beta"],
        "gpt-gamma": emb["gpt-gamma"],
    }
    if "llama-alpha" in emb:
        core["llama-alpha"] = emb["llama-alpha"]

    all_vecs = np.vstack(list(core.values()))
    all_labels = [name for name, vecs in core.items() for _ in vecs]

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    proj = reducer.fit_transform(all_vecs)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)

    for name, vecs in core.items():
        mask = [i for i, l in enumerate(all_labels) if l == name]
        x = proj[mask, 0]
        y = proj[mask, 1]
        fam = model_family(name)
        cond = condition(name)
        colour = MODEL_COLOURS[fam]
        marker = CONDITION_MARKERS[cond]
        alpha = 0.5 if cond == "null" else 0.8
        size = 20 if cond == "null" else 25
        ax.scatter(x, y, c=colour, marker=marker, s=size, alpha=alpha, zorder=5)

    # Legend: model family
    family_patches = [
        mpatches.Patch(color=MODEL_COLOURS["opus"],   label="Claude Opus"),
        mpatches.Patch(color=MODEL_COLOURS["sonnet"], label="Claude Sonnet"),
        mpatches.Patch(color=MODEL_COLOURS["gpt"],    label="GPT-4.1"),
        mpatches.Patch(color=MODEL_COLOURS["llama"],  label="Llama 3.3 70B"),
    ]
    # Legend: condition
    from matplotlib.lines import Line2D
    cond_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=GREY, markersize=7, label="Null"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=GREY, markersize=7, label="Evolved (full DMN)"),
    ]

    leg1 = ax.legend(handles=family_patches, loc="upper left",
                     facecolor=BG, edgecolor="#444", labelcolor=WHITE, fontsize=9,
                     title="Model", title_fontsize=9)
    leg1.get_title().set_color(WHITE)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=cond_handles, loc="lower left",
                     facecolor=BG, edgecolor="#444", labelcolor=WHITE, fontsize=9,
                     title="Condition", title_fontsize=9)
    leg2.get_title().set_color(WHITE)

    ax.set_title("UMAP: Session Positions in Embedding Space\n(circles = null, squares = evolved)",
                 color=WHITE, fontsize=12, pad=12)
    ax.set_xlabel("UMAP-1", color=GREY)
    ax.set_ylabel("UMAP-2", color=GREY)
    ax.tick_params(colors=GREY)
    ax.spines[:].set_color("#333")

    plt.tight_layout()
    out = FIGURES_DIR / "umap_clean.png"
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: Infrastructure effect bar chart (all 4 models) ─────────────────
def plot_infrastructure_effect():
    families = {
        "Claude Opus":    {"null": "null",        "evolved": ["alpha", "beta", "gamma"]},
        "Claude Sonnet 4.6":  {"null": "sonnet-null",  "evolved": ["sonnet-alpha", "sonnet-beta", "sonnet-gamma"]},
        "GPT-4.1":        {"null": "gpt-null",     "evolved": ["gpt-alpha", "gpt-beta", "gpt-gamma"]},
        "Llama 3.3 70B":  {"null": "llama-null",   "evolved": [k for k in ["llama-alpha", "llama-beta", "llama-gamma"] if k in emb]},
    }
    fam_colours = [MODEL_COLOURS["opus"], MODEL_COLOURS["sonnet"], MODEL_COLOURS["gpt"], MODEL_COLOURS["llama"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)

    x = np.arange(len(families))
    width = 0.35

    null_vals, evolved_vals, evolved_stds = [], [], []
    labels = []
    for fname, keys in families.items():
        null_d = mean_within_dist(emb[keys["null"]])
        ev_ds = [mean_within_dist(emb[k]) for k in keys["evolved"]]
        null_vals.append(null_d)
        evolved_vals.append(np.mean(ev_ds))
        evolved_stds.append(np.std(ev_ds))
        labels.append(fname)

    bars_null = ax.bar(x - width/2, null_vals, width, label="Null (no infrastructure)",
                       color=[c + "88" for c in fam_colours], edgecolor=fam_colours, linewidth=1.5)
    bars_evolved = ax.bar(x + width/2, evolved_vals, width, yerr=evolved_stds,
                          label="Full DMN infrastructure", color=fam_colours,
                          error_kw=dict(ecolor=WHITE, capsize=4, linewidth=1.5))

    # Label the effect size on each evolved bar
    for xi, ev, nd, col in zip(x, evolved_vals, null_vals, fam_colours):
        effect = ev - nd
        pct = effect / nd * 100
        ax.text(xi + width/2, ev + 0.015, f"+{pct:.0f}%",
                ha="center", va="bottom", color=WHITE, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=WHITE, fontsize=10)
    ax.set_ylabel("Mean within-instance cosine distance", color=GREY, fontsize=11)
    ax.set_title("Infrastructure Effect: Null vs Full DMN by Model Family\n(higher = more diverse output)",
                 color=WHITE, fontsize=12, pad=12)
    ax.tick_params(colors=GREY)
    ax.spines[:].set_color("#333")
    ax.set_ylim(0, 0.95)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.legend(facecolor=BG, edgecolor="#444", labelcolor=WHITE, fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "infrastructure_effect.png"
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: Convergence curves (key pairs only) ────────────────────────────
def plot_convergence_curves():
    window = 5

    # Only show 4 key pairs: within-Opus, within-Sonnet, within-GPT, cross-model
    key_pairs = [
        ("alpha", "beta",           "Claude Opus (α vs β)",    MODEL_COLOURS["opus"]),
        ("sonnet-alpha", "sonnet-beta", "Claude Sonnet (α vs β)", MODEL_COLOURS["sonnet"]),
        ("gpt-alpha", "gpt-beta",   "GPT-4.1 (α vs β)",        MODEL_COLOURS["gpt"]),
        ("alpha", "gpt-alpha",      "Cross-model (Opus vs GPT)", "#ffffff"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)

    for inst_a, inst_b, label, colour in key_pairs:
        if inst_a not in emb or inst_b not in emb:
            continue
        a, b = emb[inst_a], emb[inst_b]
        min_len = min(len(a), len(b))
        sims = []
        for start in range(0, min_len - window + 1):
            avg_a = np.mean(a[start:start+window], axis=0)
            avg_b = np.mean(b[start:start+window], axis=0)
            sim = 1 - cosine(avg_a, avg_b)
            sims.append((start + window//2, sim))

        x = [s[0] for s in sims]
        y = [s[1] for s in sims]
        ax.plot(x, y, color=colour, linewidth=2, label=label, alpha=0.9)
        # Trend
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), color=colour, linewidth=1, linestyle="--", alpha=0.4)

    ax.set_xlabel("Session index", color=GREY, fontsize=11)
    ax.set_ylabel("Cosine similarity (5-session window)", color=GREY, fontsize=11)
    ax.set_title("Cross-Instance Similarity Over Time\n(higher = more similar output)",
                 color=WHITE, fontsize=12, pad=12)
    ax.tick_params(colors=GREY)
    ax.spines[:].set_color("#333")
    ax.set_ylim(0, 1)
    ax.legend(facecolor=BG, edgecolor="#444", labelcolor=WHITE, fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "convergence_curves_clean.png"
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ── Figure 4: Perturbation recovery — event-aligned average ──────────────────
def plot_perturbation_recovery():
    if "perturb-no-drift" not in emb or "null" not in emb:
        print("Missing perturb-no-drift or null embeddings")
        return

    pnd_vecs = emb["perturb-no-drift"]

    # Perturbation events every 4th session (0-indexed: 3, 7, 11, ...)
    n = len(pnd_vecs)
    perturb_idx = [i for i in range(n) if (i + 1) % 4 == 0]
    non_perturb_idx = [i for i in range(n) if (i + 1) % 4 != 0]

    # Use the instance's own non-perturbation sessions as baseline centroid
    baseline_vecs = np.array([pnd_vecs[i] for i in non_perturb_idx])
    centroid = np.mean(baseline_vecs, axis=0)
    centroid /= norm(centroid)

    dists = np.array([cosine_dist(v, centroid) for v in pnd_vecs])

    # Baseline stats from non-perturbation sessions only
    baseline_dists = [dists[i] for i in non_perturb_idx]
    baseline_mean = np.mean(baseline_dists)
    baseline_std  = np.std(baseline_dists)

    # Event-aligned: collect windows of t=-1, 0, +1, +2, +3, +4 around each event
    window_before = 1
    window_after  = 5
    aligned = []
    for idx in perturb_idx:
        start = idx - window_before
        end   = idx + window_after + 1
        if start >= 0 and end <= len(dists):
            aligned.append(dists[start:end])

    aligned = np.array(aligned)  # shape: (n_events, window)
    mean_curve = np.mean(aligned, axis=0)
    std_curve  = np.std(aligned, axis=0)
    t = np.arange(-window_before, window_after + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
    fig.set_facecolor(BG)

    # LEFT: event-aligned average
    ax1.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
                alpha=0.15, color=MODEL_COLOURS["opus"])
    ax1.axhline(baseline_mean, color=MODEL_COLOURS["opus"], linewidth=1.5,
                linestyle="--", alpha=0.8, label=f"Null baseline ({baseline_mean:.3f})")
    ax1.axvline(0, color="#fd79a8", linewidth=1.5, linestyle=":", alpha=0.8,
                label="Perturbation event")

    ax1.fill_between(t, mean_curve - std_curve, mean_curve + std_curve,
                     color="#9b59b6", alpha=0.25)
    ax1.plot(t, mean_curve, color="#9b59b6", linewidth=2.5,
             label=f"Mean recovery (n={len(aligned)} events)")

    # Annotate session 0 spike and session 2 return
    ax1.annotate("Perturbation\n(session 0)", xy=(0, mean_curve[window_before]),
                 xytext=(1.2, mean_curve[window_before] + 0.04),
                 color=WHITE, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=WHITE, lw=1.2))
    ax1.annotate("~Baseline\nby session 2", xy=(2, mean_curve[window_before + 2]),
                 xytext=(3.0, mean_curve[window_before + 2] - 0.05),
                 color=WHITE, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=WHITE, lw=1.2))

    ax1.set_xlabel("Sessions relative to perturbation", color=GREY, fontsize=11)
    ax1.set_ylabel("Distance from null centroid", color=GREY, fontsize=11)
    ax1.set_title("Event-aligned average\n(±1 SD shaded)", color=WHITE, fontsize=11, pad=10)
    ax1.tick_params(colors=GREY)
    ax1.spines[:].set_color("#333")
    ax1.set_xticks(t)
    ax1.set_xticklabels([str(i) for i in t], color=GREY)
    ax1.legend(facecolor=BG, edgecolor="#444", labelcolor=WHITE, fontsize=9)

    # RIGHT: recovery time histogram
    recovery_times = []
    threshold = baseline_mean + 0.5 * baseline_std
    for idx in perturb_idx:
        recovered = None
        for offset in range(1, 6):
            if idx + offset < len(dists) and dists[idx + offset] <= threshold:
                recovered = offset
                break
        if recovered is not None:
            recovery_times.append(recovered)
        else:
            recovery_times.append(6)  # censored

    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    counts, _ = np.histogram(recovery_times, bins=bins)
    bar_labels = ["1", "2", "3", "4", "5", "6+"]
    bar_colours = [MODEL_COLOURS["opus"] if r <= 2 else "#555" for r in range(1, 7)]

    ax2.bar(range(1, 7), counts, color=bar_colours, edgecolor=BG, width=0.7)
    ax2.set_xticks(range(1, 7))
    ax2.set_xticklabels(bar_labels, color=GREY, fontsize=11)
    ax2.set_xlabel("Sessions to recover", color=GREY, fontsize=11)
    ax2.set_ylabel("Number of events", color=GREY, fontsize=11)
    ax2.set_title(f"Recovery time distribution\n(median = {int(np.median(recovery_times))} sessions, n={len(recovery_times)})",
                  color=WHITE, fontsize=11, pad=10)
    ax2.tick_params(colors=GREY)
    ax2.spines[:].set_color("#333")

    # Label bars
    for i, c in enumerate(counts):
        if c > 0:
            ax2.text(i + 1, c + 0.3, str(c), ha="center", color=WHITE, fontsize=10)

    plt.suptitle("Perturbation Recovery (drift suppressed)",
                 color=WHITE, fontsize=13, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "perturbation_recovery_clean.png"
    plt.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 5: Null distances comparison (all 4 models) ───────────────────────
def plot_null_distances():
    nulls = [
        ("Llama 3.3 70B",  "llama-null",  MODEL_COLOURS["llama"]),
        ("GPT-4.1",        "gpt-null",    MODEL_COLOURS["gpt"]),
        ("Claude Opus",    "null",        MODEL_COLOURS["opus"]),
        ("Claude Sonnet",  "sonnet-null", MODEL_COLOURS["sonnet"]),
    ]

    labels = [n[0] for n in nulls]
    vals   = [mean_within_dist(emb[n[1]]) for n in nulls]
    colours = [n[2] for n in nulls]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)

    bars = ax.barh(labels, vals, color=colours, alpha=0.85, height=0.5)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color=WHITE, fontsize=11, fontweight="bold")

    ax.axvline(1.0, color=GREY, linewidth=1, linestyle="--", alpha=0.5, label="Random baseline (≈1.0)")
    ax.set_xlabel("Mean within-instance cosine distance", color=GREY, fontsize=11)
    ax.set_title("Attractor Strength by Model\n(lower = stronger attractor, more repetitive output)",
                 color=WHITE, fontsize=12, pad=12)
    ax.tick_params(colors=GREY)
    ax.spines[:].set_color("#333")
    ax.set_xlim(0, 1.1)
    ax.legend(facecolor=BG, edgecolor="#444", labelcolor=WHITE, fontsize=10)
    for label in ax.get_yticklabels():
        label.set_color(WHITE)

    plt.tight_layout()
    out = FIGURES_DIR / "null_distances.png"
    plt.savefig(out, dpi=150, facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures...")
    print()
    plot_umap()
    plot_infrastructure_effect()
    plot_convergence_curves()
    plot_perturbation_recovery()
    plot_null_distances()
    print()
    print(f"All figures saved to: {FIGURES_DIR}")
