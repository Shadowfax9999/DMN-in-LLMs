"""
Perturbation Recovery Analysis
Research question: Is attractor recovery after perturbation driven by the drift
mechanism, or by model-intrinsic attractor pull?
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path("/Users/charliemurray/Documents/creativity work")
PERTURB_DIR       = BASE / "instances" / "perturb"       / "sessions"
NO_DRIFT_DIR      = BASE / "instances" / "perturb-no-drift" / "sessions"
OUTPUT_PNG        = BASE / "recovery_analysis.png"

# ── helpers ──────────────────────────────────────────────────────────────────
def load_sessions(session_dir: Path) -> list[tuple[str, str]]:
    """Return list of (filename, text_without_frontmatter), sorted alphabetically."""
    files = sorted(session_dir.glob("*.md"))
    sessions = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        # Strip YAML frontmatter: lines between opening --- and closing ---
        # Also handle sessions that open with a title line before ---
        text = strip_frontmatter(raw)
        sessions.append((f.name, text))
    return sessions


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter delimited by --- lines."""
    lines = text.splitlines()
    # Find first --- and second ---
    dashes = [i for i, l in enumerate(lines) if l.strip() == "---"]
    if len(dashes) >= 2:
        # Remove from first --- to second --- (inclusive)
        lines = lines[:dashes[0]] + lines[dashes[1] + 1:]
    return "\n".join(lines).strip()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def perturbation_indices(n_sessions: int) -> list[int]:
    """Return 0-indexed session indices that are perturbation sessions.
    Perturbations fire every 4th session: sessions 4, 8, 12, ... (1-indexed)
    => 0-indexed: 3, 7, 11, 15, ...
    """
    return list(range(3, n_sessions, 4))


def compute_consecutive_sims(embeddings: np.ndarray) -> np.ndarray:
    """sim[i] = cosine(embeddings[i], embeddings[i+1])"""
    sims = []
    for i in range(len(embeddings) - 1):
        sims.append(cosine_sim(embeddings[i], embeddings[i + 1]))
    return np.array(sims)


def recovery_time(sims: np.ndarray, perturb_idx: int,
                  baseline_mean: float, baseline_std: float,
                  threshold_std: float = 0.5) -> int | None:
    """
    How many steps after the perturbation does similarity return to within
    threshold_std standard deviations of baseline?
    sim[perturb_idx] is the similarity between session perturb_idx and perturb_idx+1.
    We look at sim[perturb_idx], sim[perturb_idx+1], sim[perturb_idx+2], ...
    and return how many steps until >= baseline_mean - threshold_std * baseline_std.
    Returns None if recovery is not observed within available data.
    """
    threshold = baseline_mean - threshold_std * baseline_std
    for offset in range(len(sims) - perturb_idx):
        idx = perturb_idx + offset
        if idx >= len(sims):
            return None
        if sims[idx] >= threshold:
            return offset
    return None


def post_perturb_window_sims(sims: np.ndarray, perturb_idx: int,
                              window: int = 3) -> list[float]:
    """
    Return the similarity values for the `window` steps immediately after
    a perturbation (i.e., sims[perturb_idx+1 .. perturb_idx+window]).
    These correspond to the drift-suppressed window in perturb-no-drift.
    """
    result = []
    for offset in range(1, window + 1):
        idx = perturb_idx + offset
        if idx < len(sims):
            result.append(sims[idx])
    return result


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    from sentence_transformers import SentenceTransformer

    print("=" * 70)
    print("PERTURBATION RECOVERY ANALYSIS")
    print("=" * 70)
    print()

    # Load sessions
    perturb_sessions   = load_sessions(PERTURB_DIR)
    no_drift_sessions  = load_sessions(NO_DRIFT_DIR)

    print(f"Sessions loaded:")
    print(f"  perturb:        {len(perturb_sessions)}")
    print(f"  perturb-no-drift: {len(no_drift_sessions)}")
    print()

    # Embed
    print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding perturb sessions...")
    perturb_texts    = [t for _, t in perturb_sessions]
    no_drift_texts   = [t for _, t in no_drift_sessions]

    perturb_embs   = model.encode(perturb_texts,   show_progress_bar=True,
                                   convert_to_numpy=True)
    no_drift_embs  = model.encode(no_drift_texts,  show_progress_bar=True,
                                   convert_to_numpy=True)

    print()

    # Consecutive similarities
    perturb_sims   = compute_consecutive_sims(perturb_embs)
    no_drift_sims  = compute_consecutive_sims(no_drift_embs)

    # Perturbation indices (0-indexed sessions: 3, 7, 11, ...)
    p_pidx  = perturbation_indices(len(perturb_sessions))
    nd_pidx = perturbation_indices(len(no_drift_sessions))

    # sim at perturbation is sim[perturb_idx] (transition INTO next session)
    # but we want the DROP, so we look at the sim starting at the perturb session
    # sim[i] = cosine(session_i, session_{i+1})
    # The "perturbed session" is session N (0-indexed: 3, 7, ...)
    # The drop is typically seen at sim[N] (perturbed -> next) or sim[N-1] (prev -> perturbed)
    # Since perturb fires AT session N (the session content is the perturbation),
    # the similarity drop should appear at sim[N-1] (pre-perturb -> perturb) and
    # potentially persist into sim[N], sim[N+1].
    # We'll record both sim[N-1] and sim[N] for context.

    # ── Baseline: non-perturbation consecutive similarities ──────────────────
    def baseline_sims(sims: np.ndarray, pidx_list: list[int]) -> np.ndarray:
        """Exclude transitions touching perturbation sessions."""
        perturb_set = set()
        for p in pidx_list:
            # sim[p-1]: prev -> perturb; sim[p]: perturb -> next
            for offset in [-1, 0, 1, 2, 3]:
                idx = p + offset
                if 0 <= idx < len(sims):
                    perturb_set.add(idx)
        mask = np.ones(len(sims), dtype=bool)
        mask[list(perturb_set)] = False
        return sims[mask]

    p_baseline   = baseline_sims(perturb_sims,  p_pidx)
    nd_baseline  = baseline_sims(no_drift_sims, nd_pidx)

    p_base_mean,  p_base_std  = float(np.mean(p_baseline)),  float(np.std(p_baseline))
    nd_base_mean, nd_base_std = float(np.mean(nd_baseline)), float(np.std(nd_baseline))

    print("── BASELINE SIMILARITY (non-perturbation sessions) ──────────────")
    print(f"  perturb:          mean={p_base_mean:.4f},  std={p_base_std:.4f},  n={len(p_baseline)}")
    print(f"  perturb-no-drift: mean={nd_base_mean:.4f},  std={nd_base_std:.4f},  n={len(nd_baseline)}")
    print()

    # ── Perturbation drop ────────────────────────────────────────────────────
    def drop_at(sims, pidx):
        """sim[pidx-1] is the transition into the perturbed session."""
        results = []
        for p in pidx:
            if p - 1 >= 0:
                results.append(sims[p - 1])
        return results

    p_drops  = drop_at(perturb_sims,  p_pidx)
    nd_drops = drop_at(no_drift_sims, nd_pidx)

    print("── SIMILARITY AT PERTURBATION (sim[N-1]: pre -> perturbed) ──────")
    if p_drops:
        print(f"  perturb:          mean={np.mean(p_drops):.4f},  "
              f"vs baseline {np.mean(p_drops) - p_base_mean:+.4f}  "
              f"({(np.mean(p_drops) - p_base_mean)/p_base_std:+.2f} SD)")
    if nd_drops:
        print(f"  perturb-no-drift: mean={np.mean(nd_drops):.4f},  "
              f"vs baseline {np.mean(nd_drops) - nd_base_mean:+.4f}  "
              f"({(np.mean(nd_drops) - nd_base_mean)/nd_base_std:+.2f} SD)")
    print()

    # ── Recovery times ────────────────────────────────────────────────────────
    # Recovery: how many sim-steps after sim[N-1] (the drop) until sim returns
    # to within 0.5 SD of baseline?  We start counting from sim[N-1].
    def recovery_times_for(sims, pidx, base_mean, base_std):
        times = []
        for p in pidx:
            drop_idx = p - 1  # first affected sim index
            if drop_idx < 0:
                continue
            rt = recovery_time(sims, drop_idx, base_mean, base_std, threshold_std=0.5)
            times.append(rt)
        return times

    p_rtimes  = recovery_times_for(perturb_sims,  p_pidx,  p_base_mean,  p_base_std)
    nd_rtimes = recovery_times_for(no_drift_sims, nd_pidx, nd_base_mean, nd_base_std)

    def summarise_times(times, label):
        valid = [t for t in times if t is not None]
        never = sum(1 for t in times if t is None)
        if not valid:
            print(f"  {label}: no recoveries observed (n={len(times)})")
            return
        print(f"  {label}: mean={np.mean(valid):.2f} steps, "
              f"median={np.median(valid):.1f} steps, "
              f"n={len(valid)}/{len(times)} events recovered  "
              f"(never recovered: {never})")
        print(f"           values: {valid}")

    print("── RECOVERY TIMES (steps until within 0.5 SD of baseline) ──────")
    summarise_times(p_rtimes,  "perturb         ")
    summarise_times(nd_rtimes, "perturb-no-drift")
    print()

    # ── Post-perturbation window comparison (3 sessions after each event) ────
    # This directly tests: does drift boost recovery in the 3-session window?
    def post_window_all(sims, pidx, window=3):
        all_vals = []
        for p in pidx:
            all_vals.extend(post_perturb_window_sims(sims, p - 1, window=window))
        return all_vals

    p_post  = post_window_all(perturb_sims,  p_pidx,  window=3)
    nd_post = post_window_all(no_drift_sims, nd_pidx, window=3)

    print("── POST-PERTURBATION WINDOW (3 sim steps after each event) ──────")
    print("   (In perturb-no-drift these are the drift-suppressed steps)")
    if p_post:
        print(f"  perturb:          mean sim={np.mean(p_post):.4f}  "
              f"(baseline delta: {np.mean(p_post)-p_base_mean:+.4f}, "
              f"{(np.mean(p_post)-p_base_mean)/p_base_std:+.2f} SD)")
    if nd_post:
        print(f"  perturb-no-drift: mean sim={np.mean(nd_post):.4f}  "
              f"(baseline delta: {np.mean(nd_post)-nd_base_mean:+.4f}, "
              f"{(np.mean(nd_post)-nd_base_mean)/nd_base_std:+.2f} SD)")
    if p_post and nd_post:
        diff = np.mean(p_post) - np.mean(nd_post)
        print(f"  Difference (perturb - no-drift): {diff:+.4f}")
        if diff > 0:
            print("  → drift appears to BOOST post-perturbation similarity recovery")
        elif diff < -0.005:
            print("  → drift appears to SUPPRESS post-perturbation similarity recovery")
        else:
            print("  → no meaningful difference — model-intrinsic attractor pull dominates")
    print()

    # ── Per-event detail ──────────────────────────────────────────────────────
    print("── PER-EVENT DETAIL ─────────────────────────────────────────────")
    print()
    print(f"{'Event':>5}  {'P_drop':>8}  {'P_rtime':>8}  {'ND_drop':>8}  {'ND_rtime':>8}")
    print("-" * 48)
    n_events = max(len(p_pidx), len(nd_pidx))
    for i in range(n_events):
        p_d  = f"{p_drops[i]:.4f}"  if i < len(p_drops)  else "  —  "
        p_r  = f"{p_rtimes[i]}"     if i < len(p_rtimes) else " —"
        nd_d = f"{nd_drops[i]:.4f}" if i < len(nd_drops) else "  —  "
        nd_r = f"{nd_rtimes[i]}"    if i < len(nd_rtimes)else " —"
        print(f"{i+1:>5}  {p_d:>8}  {p_r!s:>8}  {nd_d:>8}  {nd_r!s:>8}")
    print()

    # ── Interpretation ────────────────────────────────────────────────────────
    print("── INTERPRETATION ───────────────────────────────────────────────")
    print()
    p_valid  = [t for t in p_rtimes  if t is not None]
    nd_valid = [t for t in nd_rtimes if t is not None]

    if p_valid and nd_valid:
        p_mean_r  = np.mean(p_valid)
        nd_mean_r = np.mean(nd_valid)
        if p_mean_r < nd_mean_r - 0.3:
            verdict = ("Drift ACCELERATES recovery: perturb recovers faster "
                       f"({p_mean_r:.2f} steps) than perturb-no-drift "
                       f"({nd_mean_r:.2f} steps).")
        elif nd_mean_r < p_mean_r - 0.3:
            verdict = ("Drift SLOWS recovery (counter-intuitive): "
                       f"perturb-no-drift recovers faster ({nd_mean_r:.2f}) "
                       f"than perturb ({p_mean_r:.2f} steps).")
        else:
            verdict = (f"Recovery times are SIMILAR (perturb: {p_mean_r:.2f}, "
                       f"no-drift: {nd_mean_r:.2f} steps), suggesting "
                       "model-intrinsic attractor pull dominates.")
    elif not p_valid and not nd_valid:
        verdict = "Neither instance shows measurable recovery within available data."
    elif not nd_valid:
        verdict = "perturb-no-drift shows no recovery; perturb does — drift is essential."
    else:
        verdict = "perturb shows no recovery; perturb-no-drift does — drift may impede recovery."

    print(verdict)
    print()

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Colours
    C_PERTURB  = "#2196F3"   # blue
    C_NODRIFT  = "#FF9800"   # orange
    C_PERTURB_MARK  = "#D32F2F"
    C_NODRIFT_MARK  = "#BF360C"

    # ── Panel 1: time-series of similarities — perturb ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    x_p  = np.arange(len(perturb_sims))
    x_nd = np.arange(len(no_drift_sims))

    ax1.plot(x_p,  perturb_sims,  color=C_PERTURB,  lw=1.5, label="perturb",
             alpha=0.85, zorder=2)
    ax1.plot(x_nd, no_drift_sims, color=C_NODRIFT,  lw=1.5, label="perturb-no-drift",
             alpha=0.85, zorder=2)

    # Mark perturbation drop points
    for p in p_pidx:
        drop_i = p - 1
        if drop_i >= 0 and drop_i < len(perturb_sims):
            ax1.axvline(drop_i, color=C_PERTURB, lw=1.0, ls="--", alpha=0.4)
    for p in nd_pidx:
        drop_i = p - 1
        if drop_i >= 0 and drop_i < len(no_drift_sims):
            ax1.axvline(drop_i, color=C_NODRIFT, lw=1.0, ls=":", alpha=0.4)

    # Baseline bands
    ax1.axhspan(p_base_mean - 0.5 * p_base_std,
                p_base_mean + 0.5 * p_base_std,
                color=C_PERTURB, alpha=0.08, label=f"perturb baseline ±0.5SD")
    ax1.axhspan(nd_base_mean - 0.5 * nd_base_std,
                nd_base_mean + 0.5 * nd_base_std,
                color=C_NODRIFT, alpha=0.08, label=f"no-drift baseline ±0.5SD")

    ax1.set_title("Consecutive Session Similarity Over Time\n"
                  "(vertical dashed/dotted lines = perturbation drop points)",
                  fontsize=11)
    ax1.set_xlabel("Similarity index (i → i+1)")
    ax1.set_ylabel("Cosine similarity")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_ylim(0, 1)

    # ── Panel 2: recovery time histograms ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    max_t = max(
        (max(p_valid)  if p_valid  else 0),
        (max(nd_valid) if nd_valid else 0),
        4
    )
    bins = np.arange(-0.5, max_t + 1.5, 1)
    if p_valid:
        ax2.hist(p_valid,  bins=bins, color=C_PERTURB, alpha=0.7,
                 label=f"perturb (n={len(p_valid)})", edgecolor="white")
    if nd_valid:
        ax2.hist(nd_valid, bins=bins, color=C_NODRIFT, alpha=0.7,
                 label=f"no-drift (n={len(nd_valid)})", edgecolor="white")
    ax2.set_title("Recovery Time Distribution\n(steps until within 0.5 SD of baseline)", fontsize=10)
    ax2.set_xlabel("Recovery time (sim steps)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Panel 3: post-perturbation window mean sim ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    def per_event_post_window(sims, pidx, window=3):
        means = []
        for p in pidx:
            vals = post_perturb_window_sims(sims, p - 1, window=window)
            if vals:
                means.append(np.mean(vals))
        return means

    p_pw  = per_event_post_window(perturb_sims,  p_pidx)
    nd_pw = per_event_post_window(no_drift_sims, nd_pidx)

    n_events_plot = max(len(p_pw), len(nd_pw))
    ev_x = np.arange(1, n_events_plot + 1)
    if p_pw:
        ax3.plot(np.arange(1, len(p_pw) + 1), p_pw,
                 "o-", color=C_PERTURB, label="perturb", ms=6)
    if nd_pw:
        ax3.plot(np.arange(1, len(nd_pw) + 1), nd_pw,
                 "s--", color=C_NODRIFT, label="no-drift", ms=6)
    ax3.axhline(p_base_mean,  color=C_PERTURB, lw=1, ls=":", alpha=0.7,
                label=f"perturb baseline")
    ax3.axhline(nd_base_mean, color=C_NODRIFT, lw=1, ls=":", alpha=0.7,
                label=f"no-drift baseline")
    ax3.set_title("Mean Similarity in 3-Session Post-Perturbation Window\n"
                  "(per perturbation event)", fontsize=10)
    ax3.set_xlabel("Perturbation event #")
    ax3.set_ylabel("Mean cosine similarity")
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Panel 4: drop magnitude ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if p_drops:
        ax4.plot(range(1, len(p_drops) + 1),  p_drops,
                 "o-", color=C_PERTURB, label="perturb", ms=6)
    if nd_drops:
        ax4.plot(range(1, len(nd_drops) + 1), nd_drops,
                 "s--", color=C_NODRIFT, label="no-drift", ms=6)
    ax4.axhline(p_base_mean,  color=C_PERTURB, lw=1, ls=":", alpha=0.7)
    ax4.axhline(nd_base_mean, color=C_NODRIFT, lw=1, ls=":", alpha=0.7)
    ax4.set_title("Similarity at Perturbation Point\n(sim[N-1]: pre-perturb → perturbed session)", fontsize=10)
    ax4.set_xlabel("Perturbation event #")
    ax4.set_ylabel("Cosine similarity")
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 1)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Panel 5: summary bar chart ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    categories = ["Baseline\nsim", "Drop\n(perturb pt)", "Post-window\nsim",
                  "Recovery\ntime (steps)"]
    p_vals = [
        p_base_mean,
        np.mean(p_drops)  if p_drops  else 0,
        np.mean(p_post)   if p_post   else 0,
        np.mean(p_valid)  if p_valid  else 0,
    ]
    nd_vals = [
        nd_base_mean,
        np.mean(nd_drops) if nd_drops else 0,
        np.mean(nd_post)  if nd_post  else 0,
        np.mean(nd_valid) if nd_valid else 0,
    ]
    # Recovery time is on a different scale — use a secondary axis trick with
    # normalisation or just note it separately.  We'll scale recovery time
    # to [0, 1] for display only in the bar chart.
    max_rt = max(max(p_vals[3], nd_vals[3]), 1)
    p_vals_norm  = p_vals[:3]  + [p_vals[3]  / (max_rt * 5)]
    nd_vals_norm = nd_vals[:3] + [nd_vals[3] / (max_rt * 5)]

    x_bar = np.arange(len(categories))
    w = 0.35
    bars1 = ax5.bar(x_bar - w/2, p_vals_norm,  w, color=C_PERTURB, alpha=0.8,
                    label="perturb", edgecolor="white")
    bars2 = ax5.bar(x_bar + w/2, nd_vals_norm, w, color=C_NODRIFT, alpha=0.8,
                    label="no-drift", edgecolor="white")
    # Annotate actual values
    for bar, val in zip(bars1, [p_vals[0], p_vals[1], p_vals[2], p_vals[3]]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                 color=C_PERTURB)
    for bar, val in zip(bars2, [nd_vals[0], nd_vals[1], nd_vals[2], nd_vals[3]]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                 color=C_NODRIFT)
    ax5.set_xticks(x_bar)
    ax5.set_xticklabels(categories, fontsize=8)
    ax5.set_title("Summary Comparison\n(recovery time scaled for display)", fontsize=10)
    ax5.set_ylabel("Value (normalised for display)")
    ax5.legend(fontsize=8)

    fig.suptitle("Perturbation Recovery Analysis: perturb vs perturb-no-drift",
                 fontsize=13, fontweight="bold", y=0.98)

    fig.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {OUTPUT_PNG}")
    print()
    print("=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)
    print()
    print("RESEARCH QUESTION: Is attractor recovery after perturbation driven")
    print("by the drift mechanism, or by model-intrinsic attractor pull?")
    print()
    print(f"perturb sessions:        {len(perturb_sessions)}")
    print(f"perturb-no-drift sessions: {len(no_drift_sessions)}")
    print(f"Perturbation events analysed:")
    print(f"  perturb:          {len(p_pidx)} events  (indices: {p_pidx[:8]}{'...' if len(p_pidx)>8 else ''})")
    print(f"  perturb-no-drift: {len(nd_pidx)} events  (indices: {nd_pidx[:8]}{'...' if len(nd_pidx)>8 else ''})")
    print()
    print(f"Baseline similarity:")
    print(f"  perturb:          {p_base_mean:.4f} ± {p_base_std:.4f}")
    print(f"  perturb-no-drift: {nd_base_mean:.4f} ± {nd_base_std:.4f}")
    print()
    print(f"Mean similarity at perturbation drop:")
    if p_drops:
        print(f"  perturb:          {np.mean(p_drops):.4f}  "
              f"({(np.mean(p_drops)-p_base_mean)/p_base_std:+.2f} SD from baseline)")
    if nd_drops:
        print(f"  perturb-no-drift: {np.mean(nd_drops):.4f}  "
              f"({(np.mean(nd_drops)-nd_base_mean)/nd_base_std:+.2f} SD from baseline)")
    print()
    print(f"Mean similarity in 3-session post-perturbation window:")
    if p_post:
        print(f"  perturb:          {np.mean(p_post):.4f}  "
              f"({(np.mean(p_post)-p_base_mean)/p_base_std:+.2f} SD from baseline)")
    if nd_post:
        print(f"  perturb-no-drift: {np.mean(nd_post):.4f}  "
              f"({(np.mean(nd_post)-nd_base_mean)/nd_base_std:+.2f} SD from baseline)")
    print()
    print(f"Recovery time (steps to within 0.5 SD of baseline):")
    if p_valid:
        print(f"  perturb:          mean={np.mean(p_valid):.2f}, median={np.median(p_valid):.1f}  "
              f"({len(p_valid)}/{len(p_rtimes)} events recovered)")
    else:
        print(f"  perturb:          no recoveries observed ({len(p_rtimes)} events)")
    if nd_valid:
        print(f"  perturb-no-drift: mean={np.mean(nd_valid):.2f}, median={np.median(nd_valid):.1f}  "
              f"({len(nd_valid)}/{len(nd_rtimes)} events recovered)")
    else:
        print(f"  perturb-no-drift: no recoveries observed ({len(nd_rtimes)} events)")
    print()
    print("VERDICT:")
    print(verdict)
    print()


if __name__ == "__main__":
    main()
