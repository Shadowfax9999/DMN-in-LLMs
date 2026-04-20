#!/usr/bin/env python3
"""
Attractor word frequency analysis across DMN sessions.
Tests whether evolution bans actually reduce attractor term frequency.
"""

import csv
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install numpy scipy matplotlib")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────

BASE = Path("/Users/charliemurray/Documents/creativity work/instances")
OUTPUT_PLOT = Path("/Users/charliemurray/Documents/creativity work/attractor_frequency.png")

ATTRACTOR_TERMS = [
    "caulk",
    "cardinal",
    "cantaloupe",
    "tongs",
    "wednesday",
    "thursday",
    "spring peepers",
    "several",
    "fish",
    "palette",
    "phonograph",
    "theremin",
]

BAN_KEYWORDS = ["banned", "hard ban", "hard-ban", "avoid", "exhausted", "exclusion",
                "prohibited", "removed", "moved to exhausted", "do not", "never use",
                "soft-avoid", "soft avoid"]

WINDOW = 10  # sessions before/after ban to compare

# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_frontmatter(text: str) -> str:
    """Strip YAML frontmatter (between --- delimiters at file start) and
    also strip the header lines before the first --- content divider."""
    lines = text.split('\n')
    # Check for YAML frontmatter: file starts with ---
    if lines and lines[0].strip() == '---':
        # True YAML frontmatter
        end = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                end = i
                break
        if end >= 0:
            return '\n'.join(lines[end+1:])
    # Otherwise the --- is a section divider after a header; strip the header
    # (lines up to and including the first ---)
    for i, line in enumerate(lines):
        if line.strip() == '---':
            return '\n'.join(lines[i+1:])
    return text


def count_term(text: str, term: str) -> int:
    """Case-insensitive count of term occurrences in text."""
    return len(re.findall(re.escape(term), text, re.IGNORECASE))


def evolution_mentions_term(evo_text: str, term: str) -> bool:
    """Return True if the evolution file mentions the term near ban-language."""
    text_lower = evo_text.lower()
    term_lower = term.lower()
    # Find all positions where the term appears
    positions = [m.start() for m in re.finditer(re.escape(term_lower), text_lower)]
    if not positions:
        return False
    for pos in positions:
        # Look in a ±200 char window around the term for ban keywords
        window = text_lower[max(0, pos-200):pos+200+len(term_lower)]
        for kw in BAN_KEYWORDS:
            if kw in window:
                return True
    return False


def load_instance(instance_path: Path):
    """Load sessions and evolutions for one instance.
    Returns:
        sessions: list of (filename, session_index, content_stripped)
        evolutions: list of (filename, evo_index, content, timestamp_str)
    """
    sessions_dir = instance_path / "sessions"
    evos_dir = instance_path / "evolutions"

    # Load sessions sorted by filename (timestamps sort alphabetically)
    session_files = sorted(sessions_dir.glob("*.md"))
    sessions = []
    for idx, f in enumerate(session_files):
        text = f.read_text(encoding='utf-8', errors='replace')
        stripped = strip_frontmatter(text)
        sessions.append((f.name, idx, stripped))

    # Load evolutions sorted by filename
    evo_files = sorted(evos_dir.glob("*.md")) if evos_dir.exists() else []
    evolutions = []
    for idx, f in enumerate(evo_files):
        text = f.read_text(encoding='utf-8', errors='replace')
        evolutions.append((f.name, idx, text))

    return sessions, evolutions


def find_ban_session_index(evo_fname: str, sessions: list) -> int:
    """Find which session index the evolution applies after.
    The evolution timestamp should fall between two session timestamps.
    Returns the index of the first session AFTER the evolution.
    """
    # Extract timestamp prefix from evolution filename (without seconds sometimes)
    # Evolution: 2026-04-01_12-40.md  → "2026-04-01_12-40"
    # Session:   2026-04-01_12-41-12.md → "2026-04-01_12-41-12"
    evo_ts = evo_fname.replace('.md', '')

    # Normalize both to comparable strings
    # Strategy: compare lexicographically after stripping seconds from sessions
    def normalize_ts(ts):
        """Normalize to YYYY-MM-DD_HH-MM for comparison."""
        parts = ts.split('_')
        if len(parts) == 2:
            date_part = parts[0]
            time_part = parts[1]
            time_bits = time_part.split('-')
            if len(time_bits) >= 2:
                return f"{date_part}_{time_bits[0]}-{time_bits[1]}"
        return ts

    evo_norm = normalize_ts(evo_ts)

    first_after = len(sessions)  # default: after all sessions
    for sess_fname, sess_idx, _ in sessions:
        sess_ts = sess_fname.replace('.md', '')
        sess_norm = normalize_ts(sess_ts)
        if sess_norm > evo_norm:
            first_after = sess_idx
            break

    return first_after


def compute_window_freq(sessions: list, term: str, center: int, window: int, before: bool):
    """Compute total occurrences and session count in window before/after center."""
    if before:
        start = max(0, center - window)
        end = center
    else:
        start = center
        end = min(len(sessions), center + window)

    count = 0
    n_sessions = end - start
    for fname, idx, text in sessions[start:end]:
        count += count_term(text, term)
    return count, n_sessions


# ── Main analysis ──────────────────────────────────────────────────────────────

def analyse_instance(instance_path: Path):
    """Analyse one instance. Returns list of result dicts."""
    name = instance_path.name
    sessions, evolutions = load_instance(instance_path)

    if not sessions:
        return []

    results = []

    for evo_fname, evo_idx, evo_text in evolutions:
        # Find which session this evolution fires after
        ban_session = find_ban_session_index(evo_fname, sessions)

        for term in ATTRACTOR_TERMS:
            if not evolution_mentions_term(evo_text, term):
                continue

            # Compute per-session counts in before/after windows
            before_counts = []
            after_counts = []

            before_start = max(0, ban_session - WINDOW)
            for _, idx, text in sessions[before_start:ban_session]:
                before_counts.append(count_term(text, term))

            after_end = min(len(sessions), ban_session + WINDOW)
            for _, idx, text in sessions[ban_session:after_end]:
                after_counts.append(count_term(text, term))

            n_before = len(before_counts)
            n_after = len(after_counts)

            if n_before == 0 and n_after == 0:
                continue

            freq_before = sum(before_counts) / n_before if n_before > 0 else 0.0
            freq_after = sum(after_counts) / n_after if n_after > 0 else 0.0
            change = freq_after - freq_before

            # Mann-Whitney U test
            p_value = float('nan')
            if n_before >= 2 and n_after >= 2:
                try:
                    _, p_value = stats.mannwhitneyu(
                        before_counts, after_counts, alternative='two-sided'
                    )
                except Exception:
                    pass

            results.append({
                'instance': name,
                'term': term,
                'evo_file': evo_fname,
                'ban_session': ban_session,
                'n_before': n_before,
                'freq_before': freq_before,
                'n_after': n_after,
                'freq_after': freq_after,
                'change': change,
                'p_value': p_value,
                'before_counts': before_counts,
                'after_counts': after_counts,
            })

    return results


def get_term_timeseries(sessions: list, term: str):
    """Return list of per-session counts."""
    return [count_term(text, term) for _, _, text in sessions]


def get_ban_positions(sessions: list, evolutions: list, term: str):
    """Return session indices where this term was banned."""
    positions = []
    for evo_fname, evo_idx, evo_text in evolutions:
        if evolution_mentions_term(evo_text, term):
            ban_session = find_ban_session_index(evo_fname, sessions)
            positions.append(ban_session)
    return positions


# ── Select instances ───────────────────────────────────────────────────────────

def select_instances():
    """Select instances to analyse: perturb-no-drift, null, and others with 50+ sessions."""
    priority = ["perturb-no-drift", "null"]
    selected = []
    seen = set()

    for name in priority:
        p = BASE / name
        if p.exists():
            selected.append(p)
            seen.add(name)

    # Add others with 50+ sessions
    for d in sorted(BASE.iterdir()):
        if d.name in seen:
            continue
        sessions_dir = d / "sessions"
        evos_dir = d / "evolutions"
        if not sessions_dir.exists():
            continue
        n_sessions = len(list(sessions_dir.glob("*.md")))
        n_evos = len(list(evos_dir.glob("*.md"))) if evos_dir.exists() else 0
        if n_sessions >= 50 and n_evos > 0:
            selected.append(d)
            seen.add(d.name)

    return selected


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    instances = select_instances()
    print(f"\nAnalysing {len(instances)} instances:")
    for p in instances:
        n_s = len(list((p / "sessions").glob("*.md")))
        n_e = len(list((p / "evolutions").glob("*.md"))) if (p / "evolutions").exists() else 0
        print(f"  {p.name}: {n_s} sessions, {n_e} evolutions")

    all_results = []
    instance_data = {}  # name -> (sessions, evolutions)

    for inst_path in instances:
        print(f"\nLoading {inst_path.name}...")
        sessions, evolutions = load_instance(inst_path)
        instance_data[inst_path.name] = (sessions, evolutions)
        results = analyse_instance(inst_path)
        all_results.extend(results)
        print(f"  Found {len(results)} (instance, term, evolution) ban events")

    # ── Print results table ────────────────────────────────────────────────────
    print("\n" + "="*110)
    print("BAN EFFECTIVENESS TABLE")
    print("="*110)
    header = (
        f"{'Instance':<22} {'Term':<14} {'Evo':<22} "
        f"{'N_bef':>6} {'Freq_bef':>9} {'N_aft':>6} {'Freq_aft':>9} "
        f"{'Change':>8} {'p-val':>8} {'Effect':<10}"
    )
    print(header)
    print("-"*110)

    for r in sorted(all_results, key=lambda x: (x['instance'], x['term'], x['evo_file'])):
        freq_bef = r['freq_before']
        freq_aft = r['freq_after']
        change = r['change']
        p = r['p_value']
        p_str = f"{p:.3f}" if not (isinstance(p, float) and p != p) else "  n/a"
        effect = "REDUCED" if change < 0 else ("same" if change == 0 else "INCREASED")
        if change < 0 and not (isinstance(p, float) and p != p) and p < 0.05:
            effect = "REDUCED*"
        print(
            f"{r['instance']:<22} {r['term']:<14} {r['evo_file']:<22} "
            f"{r['n_before']:>6} {freq_bef:>9.3f} {r['n_after']:>6} {freq_aft:>9.3f} "
            f"{change:>+8.3f} {p_str:>8} {effect:<10}"
        )

    # ── Aggregate statistics ───────────────────────────────────────────────────
    print("\n" + "="*110)
    print("AGGREGATE STATISTICS")
    print("="*110)

    valid = [r for r in all_results if r['n_before'] > 0 and r['n_after'] > 0]
    n_total = len(valid)
    n_reduced = sum(1 for r in valid if r['change'] < 0)
    n_same = sum(1 for r in valid if r['change'] == 0)
    n_increased = sum(1 for r in valid if r['change'] > 0)
    n_sig_reduced = sum(
        1 for r in valid
        if r['change'] < 0 and not (isinstance(r['p_value'], float) and r['p_value'] != r['p_value'])
        and r['p_value'] < 0.05
    )

    print(f"\nTotal ban events with data in both windows: {n_total}")
    print(f"  Frequency reduced after ban:              {n_reduced} ({100*n_reduced/n_total:.1f}%)" if n_total else "  (no data)")
    print(f"  Frequency unchanged after ban:            {n_same} ({100*n_same/n_total:.1f}%)" if n_total else "")
    print(f"  Frequency INCREASED after ban:            {n_increased} ({100*n_increased/n_total:.1f}%)" if n_total else "")
    print(f"  Significantly reduced (p<0.05):           {n_sig_reduced} ({100*n_sig_reduced/n_total:.1f}%)" if n_total else "")

    # ── Per-term summary ───────────────────────────────────────────────────────
    print("\nPer-term summary (across all instances):")
    print(f"  {'Term':<16} {'Ban events':>10} {'% Reduced':>12} {'Avg change':>12} {'Total occ':>10}")
    print("  " + "-"*62)

    term_stats = defaultdict(list)
    term_total_occ = defaultdict(int)

    for inst_path in instances:
        inst_name = inst_path.name
        sessions, _ = instance_data[inst_name]
        for term in ATTRACTOR_TERMS:
            ts = get_term_timeseries(sessions, term)
            term_total_occ[term] += sum(ts)

    for term in ATTRACTOR_TERMS:
        term_results = [r for r in valid if r['term'] == term]
        if not term_results:
            print(f"  {term:<16} {'0':>10} {'n/a':>12} {'n/a':>12} {term_total_occ[term]:>10}")
            continue
        n = len(term_results)
        pct_red = 100 * sum(1 for r in term_results if r['change'] < 0) / n
        avg_change = sum(r['change'] for r in term_results) / n
        print(f"  {term:<16} {n:>10} {pct_red:>11.1f}% {avg_change:>+12.3f} {term_total_occ[term]:>10}")

    # ── Headline number ────────────────────────────────────────────────────────
    print("\n" + "="*110)
    if n_total > 0:
        pct = 100 * n_reduced / n_total
        print(f"HEADLINE: {pct:.1f}% of bans resulted in reduced frequency in the following {WINDOW} sessions.")
        print(f"          ({n_reduced}/{n_total} ban events showed reduction; {n_sig_reduced} were statistically significant at p<0.05)")
    else:
        print("HEADLINE: No ban events with sufficient data found.")

    # ── Find top 5 most persistent attractors for plotting ────────────────────
    # "Most persistent" = highest total occurrence count across all analysed instances
    term_occ_sorted = sorted(term_total_occ.items(), key=lambda x: x[1], reverse=True)
    top5_terms = [t for t, _ in term_occ_sorted[:5]]
    print(f"\nTop 5 most persistent attractors (by total occurrences): {top5_terms}")

    # ── Plotting ───────────────────────────────────────────────────────────────
    # Use first 3 instances for readability (perturb-no-drift, null, + one more)
    plot_instances = instances[:3]

    fig, axes = plt.subplots(
        len(top5_terms), len(plot_instances),
        figsize=(6 * len(plot_instances), 3 * len(top5_terms)),
        squeeze=False
    )
    fig.suptitle("Attractor Term Frequency Per Session\n(vertical lines = evolution ban events)", fontsize=13)

    colors = plt.cm.tab10.colors

    for col_idx, inst_path in enumerate(plot_instances):
        inst_name = inst_path.name
        sessions, evolutions = instance_data[inst_name]

        for row_idx, term in enumerate(top5_terms):
            ax = axes[row_idx][col_idx]
            ts = get_term_timeseries(sessions, term)
            ban_positions = get_ban_positions(sessions, evolutions, term)

            x = list(range(1, len(ts) + 1))
            ax.bar(x, ts, color=colors[row_idx % len(colors)], alpha=0.7, width=1.0)

            # Rolling mean
            window_size = min(10, len(ts)//3) if len(ts) > 10 else 1
            if window_size > 1:
                rolling = np.convolve(ts, np.ones(window_size)/window_size, mode='valid')
                roll_x = list(range(window_size, len(ts) + 1))
                ax.plot(roll_x, rolling, color='black', linewidth=1.2, alpha=0.6, label=f'{window_size}-sess avg')

            # Ban lines
            for bp in ban_positions:
                ax.axvline(x=bp + 0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.8)

            if row_idx == 0:
                ax.set_title(inst_name, fontsize=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'"{term}"\ncount', fontsize=9)
            ax.set_xlabel("Session #", fontsize=8)
            ax.tick_params(labelsize=7)

            total = sum(ts)
            ax.text(0.98, 0.97, f"total={total}", transform=ax.transAxes,
                    ha='right', va='top', fontsize=7, color='gray')

    # ── Save CSV: per-ban events ───────────────────────────────────────────────
    OUTPUT_PER_BAN = Path("/Users/charliemurray/Documents/creativity work/attractor_per_ban.csv")
    OUTPUT_PER_TERM = Path("/Users/charliemurray/Documents/creativity work/attractor_per_term.csv")

    per_ban_rows = []
    for r in sorted(all_results, key=lambda x: (x['instance'], x['term'], x['evo_file'])):
        p = r['p_value']
        p_str = f"{p:.6f}" if not (isinstance(p, float) and p != p) else ""
        significant = ""
        if not (isinstance(p, float) and p != p):
            significant = "yes" if p < 0.05 else "no"
        per_ban_rows.append({
            'instance': r['instance'],
            'term': r['term'],
            'evolution_session': r['evo_file'],
            'freq_before': f"{r['freq_before']:.6f}",
            'freq_after': f"{r['freq_after']:.6f}",
            'change': f"{r['change']:+.6f}",
            'p_value': p_str,
            'significant': significant,
        })

    with open(OUTPUT_PER_BAN, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'instance', 'term', 'evolution_session',
            'freq_before', 'freq_after', 'change', 'p_value', 'significant'
        ])
        writer.writeheader()
        writer.writerows(per_ban_rows)
    print(f"\nPer-ban CSV saved to: {OUTPUT_PER_BAN}")

    # ── Save CSV: per-term summary ─────────────────────────────────────────────
    per_term_rows = []
    for term in ATTRACTOR_TERMS:
        term_results = [r for r in valid if r['term'] == term]
        n = len(term_results)
        if n == 0:
            per_term_rows.append({
                'term': term,
                'ban_events': 0,
                'pct_reduced': '',
                'avg_change_per_session': '',
                'total_occurrences_all_instances': term_total_occ[term],
            })
        else:
            pct_red = 100 * sum(1 for r in term_results if r['change'] < 0) / n
            avg_change = sum(r['change'] for r in term_results) / n
            per_term_rows.append({
                'term': term,
                'ban_events': n,
                'pct_reduced': f"{pct_red:.2f}",
                'avg_change_per_session': f"{avg_change:+.6f}",
                'total_occurrences_all_instances': term_total_occ[term],
            })

    with open(OUTPUT_PER_TERM, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'term', 'ban_events', 'pct_reduced',
            'avg_change_per_session', 'total_occurrences_all_instances'
        ])
        writer.writeheader()
        writer.writerows(per_term_rows)
    print(f"Per-term CSV saved to: {OUTPUT_PER_TERM}")

    # Legend
    red_line = mpatches.Patch(color='red', alpha=0.8, label='Evolution ban')
    fig.legend(handles=[red_line], loc='lower right', fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(OUTPUT_PLOT, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {OUTPUT_PLOT}")

    # ── Per-instance breakdown ─────────────────────────────────────────────────
    print("\n" + "="*110)
    print("PER-INSTANCE BREAKDOWN")
    print("="*110)
    for inst_path in instances:
        inst_name = inst_path.name
        sessions, evolutions = instance_data[inst_name]
        inst_results = [r for r in valid if r['instance'] == inst_name]
        if not inst_results:
            print(f"\n{inst_name}: no ban events with data")
            continue
        n_r = sum(1 for r in inst_results if r['change'] < 0)
        pct = 100 * n_r / len(inst_results)
        print(f"\n{inst_name}: {len(inst_results)} ban events — {pct:.1f}% reduced")

        # Show term occurrences per session
        print(f"  Term occurrence rates (occ/session across all {len(sessions)} sessions):")
        for term in ATTRACTOR_TERMS:
            ts = get_term_timeseries(sessions, term)
            total = sum(ts)
            if total > 0:
                rate = total / len(sessions)
                n_nonzero = sum(1 for c in ts if c > 0)
                print(f"    {term:<16}: {total:>4} total occ, {rate:.3f}/session, present in {n_nonzero}/{len(sessions)} sessions")


if __name__ == "__main__":
    main()
