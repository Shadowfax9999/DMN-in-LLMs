#!/usr/bin/env python3
"""Build an HTML dashboard for the DMN project.

Reads sessions, evolutions, state, and program.md to generate
a self-contained HTML file with embedded data and charts.
"""

import base64
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

DIR = Path(__file__).parent
SESSIONS_DIR = DIR / "sessions"
EVOLUTIONS_DIR = DIR / "evolutions"
STATE_FILE = DIR / ".dmn_state.json"
PROGRAM_FILE = DIR / "program.md"
CLAUDE_FILE = DIR / "CLAUDE.md"
OUTPUT_FILE = DIR / "dashboard.html"

# ── Common stop words to exclude from theme extraction ────────────────────────
STOP_WORDS = set("""
the a an and or but in on at to for of is it its it's that this these those
was were be been being have has had do does did will would shall should may
might can could not no nor so if then than too very just about above after
again all also am any are as back because before between both by came come
could day did different do each end even find first from get give go going
gone got great had has have her here him his how i if in into is its just
know large last let like line long look made make man many may me more most
much must my name never new next no not now number of off old on one only
or other our out over own part people place point right said same saw say
second see she side small so some something state still such take tell than
that the their them then there these they thing think this those thought
three through time to together too two under up us use very want water way
we well went were what when where which while who why will with word work
world would write year you your been from with they them what when where
which while into through about before after between under over again once
during each every some any most other another such only also back well just
still even how much many those these here there where does don't doesn't
didn't wasn't weren't won't wouldn't couldn't shouldn't aren't isn't hasn't
haven't hadn't let's that's what's who's it's i'm you're we're they're he's
she's there's here's where's how's i've you've we've they've i'll you'll
we'll they'll i'd you'd we'd they'd like one two way down because though
almost already always never sometimes often usually gets got went goes come
came makes made says said knows knew things thing thought think whether
without within upon until quite rather really simply actually perhaps maybe
seems seemed become became keep kept kind sort begin began whole another
""".split())


def parse_sessions(sessions_dir=None):
    """Read all session files and extract metadata."""
    if sessions_dir is None:
        sessions_dir = SESSIONS_DIR
    sessions = []
    for f in sorted(sessions_dir.glob("*.md")):
        text = f.read_text()
        lines = text.strip().split("\n")

        # Parse session number from first line: "# 42"
        num = None
        if lines and lines[0].startswith("# "):
            try:
                num = int(lines[0][2:].strip())
            except ValueError:
                pass

        # Parse seed line
        concept = None
        seed_line = ""
        for line in lines[:5]:
            if line.startswith("*seed:"):
                seed_line = line
                # Extract concept if present (last item after ·)
                parts = line.split("·")
                if len(parts) >= 3:
                    concept = parts[-1].strip().rstrip("*").strip()
                break

        # Get the body text (after the --- separator)
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip() == "---" and i > 0:
                body_start = i + 1
                break
        body = "\n".join(lines[body_start:]).strip()

        # Count section breaks
        section_breaks = body.count("\n---\n") + body.count("\n---")

        # Word count
        words = body.split()
        word_count = len(words)

        # Detect ending type
        ending_type = "complete"
        if body:
            last_chars = body.rstrip()[-3:] if len(body.rstrip()) >= 3 else body.rstrip()
            if last_chars.endswith("—") or last_chars.endswith("–"):
                ending_type = "mid-sentence"
            elif last_chars.endswith("...") or last_chars.endswith("…"):
                ending_type = "trailing"
            elif not body.rstrip()[-1] in ".!?\"'":
                ending_type = "mid-sentence"

        # Extract key themes (top distinctive words)
        word_freq = Counter()
        for w in words:
            w_clean = re.sub(r"[^a-z']", "", w.lower())
            if w_clean and len(w_clean) > 3 and w_clean not in STOP_WORDS:
                word_freq[w_clean] += 1
        themes = [w for w, _ in word_freq.most_common(15)]

        # Parse date from filename
        date_str = f.stem  # e.g. "2026-03-25_16-56"
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d_%H-%M")
        except ValueError:
            try:
                dt = datetime.strptime(date_str.rsplit("_", 1)[0], "%Y-%m-%d")
            except ValueError:
                dt = None

        # HTML-safe body for rendering
        import html as html_mod
        body_html = html_mod.escape(body).replace("\n---\n", '\n<hr class="session-break">\n').replace("\n", "<br>\n")

        sessions.append({
            "num": num,
            "file": f.name,
            "date": dt.isoformat() if dt else None,
            "date_display": dt.strftime("%b %d, %H:%M") if dt else f.stem,
            "concept": concept,
            "word_count": word_count,
            "section_breaks": section_breaks,
            "ending_type": ending_type,
            "themes": themes,
            "first_line": body.split("\n")[0][:120] if body else "",
            "body": body_html,
            "seed_line": seed_line,
        })

    return sessions


def parse_evolutions(evolutions_dir=None):
    """Read all evolution files and extract metadata."""
    if evolutions_dir is None:
        evolutions_dir = EVOLUTIONS_DIR
    evolutions = []
    for f in sorted(evolutions_dir.glob("*.md")):
        if "_raw" in f.name:
            continue  # skip raw/failed parses
        text = f.read_text()

        # Parse date from filename
        date_str = f.stem  # e.g. "2026-03-25_17-16"
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d_%H-%M")
        except ValueError:
            dt = None

        # Extract reflection section
        reflection = ""
        changes = ""
        in_reflection = False
        in_changes = False
        for line in text.split("\n"):
            if line.startswith("## Reflection"):
                in_reflection = True
                in_changes = False
                continue
            elif line.startswith("## Changes") or line.startswith("## CLAUDE"):
                in_reflection = False
                if "Changes" in line:
                    in_changes = True
                continue
            elif line.startswith("## "):
                in_reflection = False
                in_changes = False
                continue
            if in_reflection:
                reflection += line + "\n"
            if in_changes:
                changes += line + "\n"

        evolutions.append({
            "file": f.name,
            "date": dt.isoformat() if dt else None,
            "date_display": dt.strftime("%b %d, %H:%M") if dt else f.stem,
            "reflection": reflection.strip(),
            "changes": changes.strip(),
        })

    return evolutions


def parse_concept_bank(program_file=None):
    """Extract current concept bank from program.md."""
    if program_file is None:
        program_file = PROGRAM_FILE
    if not program_file.exists():
        return [], []

    text = program_file.read_text()

    # Find concept bank section
    concepts = []
    in_bank = False
    for line in text.split("\n"):
        if "concept bank" in line.lower() or "## concepts" in line.lower():
            in_bank = True
            continue
        if in_bank and line.startswith("## "):
            break
        if in_bank and line.strip():
            # Extract italicized or plain concepts
            found = re.findall(r"\*([^*]+)\*", line)
            if found:
                for group in found:
                    concepts.extend([c.strip() for c in group.split(",")])
            else:
                # Try comma-separated plain text
                if "," in line and not line.startswith("#"):
                    concepts.extend([c.strip() for c in line.split(",") if c.strip()])

    # Extract exhausted themes
    exhausted = []
    for line in text.split("\n"):
        if "exhausted" in line.lower() and "theme" in line.lower():
            found = re.findall(r'"([^"]+)"', line)
            if not found:
                found = re.findall(r"\*([^*]+)\*", line)
            exhausted.extend(found)

    return concepts, exhausted


def parse_reflections(claude_file=None):
    """Extract reflections from CLAUDE.md."""
    if claude_file is None:
        claude_file = CLAUDE_FILE
    if not claude_file.exists():
        return []

    text = claude_file.read_text()
    reflections = []

    for line in text.split("\n"):
        if line.startswith("**2026-"):
            # Extract date and text
            match = re.match(r"\*\*(\d{4}-\d{2}-\d{2})\*\*:?\s*(.*)", line)
            if match:
                reflections.append({
                    "date": match.group(1),
                    "text": match.group(2)[:200],
                })

    return reflections


def build_concept_usage(sessions):
    """Track which concepts were injected and when."""
    usage = {}
    for s in sessions:
        if s["concept"]:
            c = s["concept"]
            if c not in usage:
                usage[c] = []
            usage[c].append(s["num"])
    return usage


def build_theme_timeline(sessions):
    """Build theme frequency over rolling windows of 5 sessions."""
    if len(sessions) < 5:
        return []

    timeline = []
    for i in range(0, len(sessions), 5):
        window = sessions[i:i+5]
        if not window:
            continue
        all_themes = Counter()
        for s in window:
            for t in s["themes"][:8]:
                all_themes[t] += 1
        timeline.append({
            "sessions": f"{window[0]['num']}–{window[-1]['num']}",
            "top_themes": [{"word": w, "count": c} for w, c in all_themes.most_common(10)],
        })

    return timeline


def load_analysis_data():
    """Load analysis results and encode plot images as base64."""
    analysis_dir = DIR / "analysis"
    analysis = {"available": False, "images": {}, "results": {}}

    results_file = analysis_dir / "results.json"
    if not results_file.exists():
        return analysis

    analysis["available"] = True
    analysis["results"] = json.loads(results_file.read_text())

    # Encode plot PNGs as base64
    for name in ["umap_attractor_map", "convergence_curves", "distance_over_time", "entropy_over_time"]:
        img_path = analysis_dir / f"{name}.png"
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            analysis["images"][name] = f"data:image/png;base64,{b64}"

    return analysis


def generate_html(sessions, evolutions, concept_bank, exhausted, concept_usage,
                  theme_timeline, reflections, state, instances=None):
    """Generate the full dashboard HTML with tabbed layout."""

    analysis = load_analysis_data()

    def embed_figure(filename):
        """Embed a PNG figure as a base64 img tag."""
        fig_path = DIR / "analysis" / filename
        if fig_path.exists():
            b64 = base64.b64encode(fig_path.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;" />'
        return '<p style="color:#555;">Figure not yet generated. Run <code>python3 make_figures.py</code></p>'

    data = {
        "sessions": sessions,
        "evolutions": evolutions,
        "conceptBank": concept_bank,
        "exhaustedThemes": exhausted,
        "conceptUsage": concept_usage,
        "themeTimeline": theme_timeline,
        "reflections": reflections,
        "state": state,
        "instances": instances or {},
        "analysis": analysis,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DMN Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f;
    color: #c8c8d0;
    line-height: 1.6;
    height: 100vh;
    overflow: hidden;
}}

/* ── Nav bar ────────────────────────────────────────────────────── */
.navbar {{
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 0.75rem 2rem;
    background: #0d0d14;
    border-bottom: 1px solid #1a1a2a;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 200;
}}
.navbar-title {{
    font-size: 0.85rem;
    font-weight: 300;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a5a6a;
    white-space: nowrap;
}}
.navbar-tabs {{
    display: flex;
    gap: 0;
}}
.nav-tab {{
    padding: 0.5rem 1.2rem;
    font-size: 0.8rem;
    color: #5a5a6a;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}}
.nav-tab:hover {{ color: #9a9aaa; }}
.nav-tab.active {{
    color: #c8c8d0;
    border-bottom-color: #6a6aaa;
}}
.navbar-stats {{
    margin-left: auto;
    font-size: 0.72rem;
    color: #3a3a4a;
    white-space: nowrap;
}}

/* ── Tab content ────────────────────────────────────────────────── */
.tab-content {{
    display: none !important;
    margin-top: 52px;
    height: calc(100vh - 52px);
}}
.tab-content.active {{ display: flex !important; }}

/* ── Sessions tab ───────────────────────────────────────────────── */

.session-list {{
    width: 320px;
    min-width: 320px;
    height: calc(100vh - 52px);
    overflow-y: auto;
    border-right: 1px solid #1a1a2a;
    background: #0d0d14;
}}
.session-item {{
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #111118;
    cursor: pointer;
    transition: background 0.15s;
}}
.session-item:hover {{ background: #12121e; }}
.session-item.active {{ background: #16162a; border-left: 3px solid #6a6aaa; }}
.session-item-num {{
    font-size: 0.82rem;
    color: #b8b8c8;
    font-weight: 500;
}}
.session-item-meta {{
    font-size: 0.7rem;
    color: #4a4a5a;
    margin-top: 0.1rem;
}}
.session-item-preview {{
    font-size: 0.72rem;
    color: #5a5a6a;
    margin-top: 0.2rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.concept-pill {{
    display: inline-block;
    font-size: 0.65rem;
    padding: 0.05rem 0.4rem;
    border-radius: 8px;
    background: #1a2a3a;
    color: #6a9aba;
    margin-left: 0.3rem;
}}

.session-reader {{
    flex: 1;
    height: calc(100vh - 52px);
    overflow-y: auto;
    padding: 2rem 3rem;
    max-width: 800px;
}}
.reader-num {{
    font-size: 1.8rem;
    font-weight: 200;
    color: #e8e8f0;
}}
.reader-meta {{
    font-size: 0.78rem;
    color: #5a5a6a;
    margin: 0.3rem 0 1.5rem;
}}
.reader-body {{
    font-size: 0.92rem;
    line-height: 1.8;
    color: #b8b8c8;
}}
.reader-body hr.session-break {{
    border: none;
    border-top: 1px solid #1a1a2a;
    margin: 1.5rem 0;
}}
.reader-themes {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1a1a2a;
}}
.reader-empty {{
    color: #3a3a4a;
    font-style: italic;
    padding: 4rem 2rem;
    text-align: center;
}}

/* ── Evolutions tab ─────────────────────────────────────────────── */
#tab-evolutions {{
    flex-direction: column;
    overflow-y: auto;
    padding: 2rem 3rem;
    max-width: 900px;
}}
#tab-evolutions .evo-entry {{
    flex-shrink: 0;
}}
.evo-entry {{
    margin: 0.75rem 0;
    background: #0f0f18;
    border-left: 2px solid #cc6644;
    border-radius: 0 6px 6px 0;
    overflow: hidden;
}}
.evo-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.2rem;
    cursor: pointer;
    transition: background 0.15s;
}}
.evo-header:hover {{ background: #18182a; }}
.evo-date {{
    font-size: 0.75rem;
    color: #e08855;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
    font-weight: 500;
}}
.evo-preview {{
    font-size: 0.82rem;
    color: #9a9aaa;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
}}
.evo-chevron {{
    color: #6a6a7a;
    font-size: 0.8rem;
    transition: transform 0.2s;
}}
.evo-entry.open .evo-chevron {{ transform: rotate(90deg); }}
.evo-body {{
    display: none;
    padding: 0 1.2rem 1.2rem;
    font-size: 0.85rem;
    color: #a8a8b8;
    line-height: 1.8;
}}
.evo-entry.open .evo-body {{ display: block; }}

/* ── Instances tab ──────────────────────────────────────────────── */
#tab-instances {{
    flex-direction: column;
    overflow-y: auto;
    padding: 2rem;
}}
.instances-columns {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-bottom: 2rem;
}}
.instance-col {{
    background: #0f0f18;
    border: 1px solid #1a1a2a;
    border-radius: 8px;
    overflow: hidden;
}}
.instance-header {{
    padding: 1rem;
    border-bottom: 1px solid #1a1a2a;
}}
.instance-name {{
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 0.3rem;
}}
.instance-name.alpha {{ color: #8ab4f8; }}
.instance-name.beta {{ color: #a8d8a8; }}
.instance-name.gamma {{ color: #d8a8d8; }}
.instance-stats {{
    font-size: 0.75rem;
    color: #5a5a6a;
}}
.instance-latest {{
    padding: 1rem;
    font-size: 0.82rem;
    color: #8a8a9a;
    line-height: 1.7;
    max-height: 400px;
    overflow-y: auto;
}}
.instance-latest-label {{
    font-size: 0.68rem;
    color: #4a4a5a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}}
.instance-latest hr.session-break {{
    border: none;
    border-top: 1px solid #1a1a2a;
    margin: 1rem 0;
}}

.theme-compare {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
}}
.theme-compare-col {{
    background: #0f0f18;
    border: 1px solid #1a1a2a;
    border-radius: 8px;
    padding: 1rem;
}}
.theme-compare-col h4 {{
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}}
.theme-compare-col h4.alpha {{ color: #8ab4f8; }}
.theme-compare-col h4.beta {{ color: #a8d8a8; }}
.theme-compare-col h4.gamma {{ color: #d8a8d8; }}
.theme-tag {{
    font-size: 0.73rem;
    padding: 0.12rem 0.45rem;
    border-radius: 3px;
    background: #1a1a2a;
    color: #8a8aaa;
    display: inline-block;
    margin: 0.1rem;
}}
.theme-tag.hot {{ background: #2a2a5a; color: #aaaacc; }}
.theme-tag.shared {{ background: #2a3a2a; color: #8aba8a; }}

/* ── Overview tab ───────────────────────────────────────────────── */
#tab-overview {{
    flex-direction: column;
    overflow-y: auto;
    padding: 2rem 3rem;
}}
#tab-overview > * {{
    flex-shrink: 0;
}}
#tab-overview h2 {{
    font-size: 1rem;
    font-weight: 500;
    color: #8a8a9a;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1a2a;
}}
.stats {{
    display: flex;
    gap: 2rem;
    margin-bottom: 1.5rem;
}}
.stat-num {{ font-size: 2rem; font-weight: 200; color: #e8e8f0; }}
.stat-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #5a5a6a; }}

.timeline {{ overflow-x: auto; padding: 1rem 0 2rem; }}
.timeline-track {{
    display: flex;
    align-items: flex-end;
    gap: 2px;
    min-width: max-content;
    height: 120px;
}}
.session-bar {{
    width: 10px;
    min-height: 4px;
    background: #2a2a4a;
    border-radius: 2px 2px 0 0;
    cursor: pointer;
    transition: background 0.2s;
    position: relative;
}}
.session-bar:hover {{ background: #5a5aaa; }}
.session-bar.has-concept {{ background: #3a3a6a; }}
.evolution-marker {{
    width: 2px;
    height: 120px;
    background: #cc6644;
    position: relative;
    flex-shrink: 0;
    opacity: 0.7;
}}
.form-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, 24px);
    gap: 2px;
    margin: 1rem 0;
}}
.form-cell {{
    width: 22px;
    height: 22px;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.55rem;
    color: #4a4a5a;
}}
.ending-complete {{ background: #1a2a1a; }}
.ending-mid-sentence {{ background: #2a1a2a; }}
.ending-trailing {{ background: #2a2a1a; }}
.legend {{ display: flex; gap: 1.5rem; margin: 0.5rem 0; font-size: 0.75rem; color: #5a5a6a; }}
.legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
.legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; }}

.concept-grid {{ display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 1rem 0; }}
.concept-chip {{
    font-size: 0.72rem;
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
    border: 1px solid #1a1a2a;
}}
.concept-used {{ background: #1a2a3a; color: #6a9aba; border-color: #2a3a4a; }}
.concept-unused {{ background: #0f0f14; color: #3a3a4a; border-color: #1a1a20; }}
.concept-exhausted {{ background: #2a1a1a; color: #8a4a4a; border-color: #3a2a2a; text-decoration: line-through; }}

.theme-window {{
    display: inline-flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin: 0.5rem 0 1rem;
    padding: 0.75rem;
    background: #0f0f18;
    border-radius: 4px;
    border: 1px solid #1a1a2a;
    width: 100%;
}}
.theme-window-label {{
    font-size: 0.7rem;
    color: #4a4a5a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    width: 100%;
    margin-bottom: 0.3rem;
}}

/* ── Analysis tab ──────────────────────────────────────────────── */
#tab-analysis {{
    flex-direction: column;
    overflow-y: auto;
    padding: 0;
}}
.analysis-container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 3rem;
}}
.analysis-header {{
    margin-bottom: 2rem;
}}
.analysis-header h2 {{
    font-size: 1.2rem;
    font-weight: 400;
    color: #e8e8f0;
    margin-bottom: 0.3rem;
}}
.analysis-subtitle {{
    font-size: 0.8rem;
    color: #5a5a6a;
    margin-bottom: 1.5rem;
}}
.analysis-summary {{
    display: flex;
    gap: 2rem;
    padding: 1rem 1.5rem;
    background: #0f0f18;
    border: 1px solid #1a1a2a;
    border-radius: 6px;
    margin-bottom: 1rem;
}}
.analysis-stat {{
    text-align: center;
}}
.analysis-stat-val {{
    font-size: 1.8rem;
    font-weight: 200;
    color: #e8e8f0;
}}
.analysis-stat-val.converging {{ color: #2ecc71; }}
.analysis-stat-val.diverging {{ color: #e74c3c; }}
.analysis-stat-label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5a5a6a;
}}
.analysis-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
}}
.analysis-card {{
    background: #0f0f18;
    border: 1px solid #1a1a2a;
    border-radius: 6px;
    padding: 1.25rem;
}}
.analysis-card-wide {{
    grid-column: 1 / -1;
}}
.analysis-card h3 {{
    font-size: 0.85rem;
    font-weight: 500;
    color: #9a9aaa;
    margin-bottom: 0.4rem;
}}
.analysis-desc {{
    font-size: 0.72rem;
    color: #4a4a5a;
    margin-bottom: 1rem;
    line-height: 1.5;
}}
.analysis-img-wrap {{
    text-align: center;
}}
.analysis-img-wrap img {{
    max-width: 100%;
    border-radius: 4px;
}}
.analysis-note {{
    font-size: 0.72rem;
    color: #3a3a4a;
    padding: 1rem;
    border-top: 1px solid #1a1a2a;
}}
.analysis-note code {{
    background: #1a1a2a;
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    font-size: 0.7rem;
}}
.analysis-empty {{
    text-align: center;
    padding: 4rem 2rem;
    color: #3a3a4a;
    font-size: 0.85rem;
}}
.analysis-filter {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.8rem 0;
}}
.filter-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.6rem;
    border-radius: 12px;
    font-size: 0.78rem;
    cursor: pointer;
    border: 1px solid #2a2a3a;
    background: #1a1a2a;
    color: #8888aa;
    transition: all 0.15s;
    user-select: none;
}}
.filter-chip.active {{
    border-color: var(--chip-color, #e74c3c);
    color: var(--chip-color, #e74c3c);
    background: rgba(255,255,255,0.05);
}}
.filter-chip input {{
    display: none;
}}
.filter-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--chip-color, #888);
}}
</style>
</head>
<body>

<!-- Nav bar -->
<div class="navbar">
    <div class="navbar-title">DMN</div>
    <div class="navbar-tabs">
        <div class="nav-tab active" data-tab="overview" onclick="switchTab('overview')">Overview</div>
        <div class="nav-tab" data-tab="sessions" onclick="switchTab('sessions')">Sessions</div>
        <div class="nav-tab" data-tab="evolutions" onclick="switchTab('evolutions')">Evolutions</div>
        <div class="nav-tab" data-tab="instances" onclick="switchTab('instances')">Instances</div>
        <div class="nav-tab" data-tab="analysis" onclick="switchTab('analysis')">Analysis</div>
    </div>
    <div class="navbar-stats" id="navbar-stats"></div>
</div>

<!-- Sessions tab -->
<div class="tab-content" id="tab-sessions">
    <div class="session-list" id="session-list"></div>
    <div class="session-reader" id="session-reader">
        <div class="reader-empty">Select a session to read</div>
    </div>
</div>

<!-- Evolutions tab -->
<div class="tab-content" id="tab-evolutions"></div>

<!-- Instances tab -->
<div class="tab-content" id="tab-instances"></div>

<!-- Overview tab -->
<div class="tab-content active" id="tab-overview"></div>

<!-- Analysis tab -->
<div class="tab-content" id="tab-analysis">
    <div class="analysis-container">
        <div class="analysis-header">
            <h2>Attractor Analysis</h2>
            <p class="analysis-subtitle">Embedding-based analysis of session trajectories across instances</p>
            <div class="analysis-filter" id="analysis-filter">
                <span style="color:#8888aa;font-size:0.85rem;margin-right:0.5rem;">Compare:</span>
            </div>
            <div class="analysis-summary" id="analysis-summary"></div>
        </div>
        <div class="analysis-grid">
            <div class="analysis-card analysis-card-wide">
                <h3>Attractor Map (UMAP)</h3>
                <p class="analysis-desc">Each dot is a session embedded in high-dimensional space and projected to 2D. Lines show trajectories through idea-space. Use the checkboxes above to filter instances.</p>
                <div class="analysis-img-wrap" style="height:500px;" id="chart-umap">
                    <canvas id="umap-canvas"></canvas>
                </div>
            </div>
            <div class="analysis-card analysis-card-wide">
                <h3>Cross-Instance Convergence</h3>
                <p class="analysis-desc">Cosine similarity between instance pairs over 5-session sliding windows. Lower = more similar. Use the checkboxes above to filter which instances to compare.</p>
                <div class="analysis-img-wrap" style="height:400px;" id="chart-convergence">
                    <canvas id="convergence-canvas"></canvas>
                </div>
            </div>
            <div class="analysis-card analysis-card-wide">
                <h3>Embedding Variance Over Time</h3>
                <p class="analysis-desc">Rolling variance of session embeddings within 5-session windows. Higher = more diverse output. Drops suggest the system settling into a groove. Spikes often follow evolution interventions.</p>
                <div class="analysis-img-wrap" style="height:400px;" id="chart-variance">
                    <canvas id="variance-canvas"></canvas>
                </div>
            </div>
        </div>
        <div class="analysis-card analysis-card-wide">
                <h3>Perturbation Recovery</h3>
                <p class="analysis-desc">Distance from attractor centroid over time. Red lines mark perturbation events (every 4th session). Blue band shows baseline ± 1σ. The system snaps back within 1 session — a steep-sided attractor.</p>
                <div class="analysis-img-wrap">{embed_figure("perturbation_recovery.png")}</div>
            </div>
            <div class="analysis-card analysis-card-wide">
                <h3>Null Baseline vs Controls</h3>
                <p class="analysis-desc">Within-instance and cross-instance distances. Null sessions (no drift, no evolution) are far more self-similar (0.494) than controls (~0.73), but sit in the same embedding region. The attractor is in the weights.</p>
                <div class="analysis-img-wrap">{embed_figure("null_comparison.png")}</div>
            </div>
            <div class="analysis-card analysis-card-wide">
                <h3>Convergence by Phase</h3>
                <p class="analysis-desc">Mean cosine distance between control pairs across 20-session windows. Green = closer, red = further. Shows the convergence-divergence oscillation driven by evolution interventions.</p>
                <div class="analysis-img-wrap">{embed_figure("phase_heatmap.png")}</div>
            </div>
        </div>
        <div class="analysis-note">
            <p>Generated by <code>analyse.py</code> using sentence-transformers (all-MiniLM-L6-v2) embeddings, UMAP projection, and cosine similarity metrics. Re-run <code>python3 analyse.py</code> to update.</p>
        </div>
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>

<script>
const DATA = {json.dumps(data, indent=None)};

// ── Tab switching ────────────────────────────────────────────────
function switchTab(name) {{
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    document.querySelector(`.nav-tab[data-tab="${{name}}"]`).classList.add('active');
}}

// ── Navbar stats ────────────────────────────────────────────────
const totalWords = DATA.sessions.reduce((a, s) => a + s.word_count, 0);
document.getElementById('navbar-stats').textContent =
    `${{DATA.sessions.length}} sessions \u00b7 ${{DATA.evolutions.length}} evolutions \u00b7 ${{totalWords.toLocaleString()}} words`;

// ── Sessions tab ────────────────────────────────────────────────
const sessionList = document.getElementById('session-list');
const sessionReader = document.getElementById('session-reader');

// Build list (newest first)
const sortedSessions = [...DATA.sessions].reverse();

sortedSessions.forEach((s, i) => {{
    const item = document.createElement('div');
    item.className = 'session-item' + (i === 0 ? ' active' : '');
    item.dataset.idx = DATA.sessions.length - 1 - i;
    item.innerHTML = `
        <div class="session-item-num">
            #${{s.num}}
            ${{s.concept ? '<span class="concept-pill">' + s.concept + '</span>' : ''}}
        </div>
        <div class="session-item-meta">${{s.date_display}} \u00b7 ${{s.word_count}} words \u00b7 ${{s.ending_type}}</div>
        <div class="session-item-preview">${{s.first_line}}</div>
    `;
    item.onclick = () => showSession(s, item);
    sessionList.appendChild(item);
}});

function showSession(s, clickedItem) {{
    // Update active state
    document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
    if (clickedItem) clickedItem.classList.add('active');

    sessionReader.innerHTML = `
        <div class="reader-num">#${{s.num}}</div>
        <div class="reader-meta">
            ${{s.date_display}} \u00b7 ${{s.word_count}} words \u00b7 ${{s.section_breaks}} section breaks \u00b7 ${{s.ending_type}} ending<br>
            ${{s.seed_line || ''}}
        </div>
        <div class="reader-body">${{s.body || '<em>No body text available</em>'}}</div>
        <div class="reader-themes">
            ${{s.themes.slice(0, 12).map(t => '<span class="theme-tag">' + t + '</span>').join('')}}
        </div>
    `;
    sessionReader.scrollTop = 0;
}}

// Show newest session by default
if (sortedSessions.length > 0) {{
    showSession(sortedSessions[0], document.querySelector('.session-item'));
}}

// ── Evolutions tab ──────────────────────────────────────────────
const evoTab = document.getElementById('tab-evolutions');
const INST_COLORS = {{
    main: '#e08855', alpha: '#e74c3c', beta: '#3498db', gamma: '#2ecc71',
    replay: '#f39c12', perturb: '#9b59b6', switch: '#1abc9c'
}};

// Gather all evolutions from main + instances
const allEvos = [];
DATA.evolutions.forEach(e => allEvos.push({{ ...e, instance: 'main' }}));
for (const [name, inst] of Object.entries(DATA.instances)) {{
    if (inst.evolutions) {{
        inst.evolutions.forEach(e => allEvos.push({{ ...e, instance: name }}));
    }}
}}
// Sort by date descending
allEvos.sort((a, b) => (b.date || '').localeCompare(a.date || ''));

allEvos.forEach(e => {{
    const entry = document.createElement('div');
    entry.className = 'evo-entry';
    const preview = e.reflection.substring(0, 100).replace(/\\n/g, ' ');
    const color = INST_COLORS[e.instance] || '#888';
    const label = e.instance !== 'main' ? `<span style="color:${{color}};font-weight:600;margin-right:0.5rem;font-size:0.75rem;text-transform:uppercase">${{e.instance}}</span>` : `<span style="color:${{color}};font-weight:600;margin-right:0.5rem;font-size:0.75rem;text-transform:uppercase">main</span>`;
    entry.style.borderLeftColor = color;
    entry.innerHTML = `
        <div class="evo-header" onclick="this.parentElement.classList.toggle('open')">
            <div class="evo-date">${{label}}${{e.date_display}}</div>
            <div class="evo-preview">${{preview}}\u2026</div>
            <div class="evo-chevron">\u25b6</div>
        </div>
        <div class="evo-body">${{e.reflection.replace(/\\n/g, '<br>')}}</div>
    `;
    evoTab.appendChild(entry);
}});

// ── Instances tab ───────────────────────────────────────────────
const instTab = document.getElementById('tab-instances');
const instNames = Object.keys(DATA.instances);
const instColors = {{ alpha: '#8ab4f8', beta: '#a8d8a8', gamma: '#d8a8d8' }};

if (instNames.length > 0) {{
    // Columns with latest session
    const colsDiv = document.createElement('div');
    colsDiv.className = 'instances-columns';

    instNames.forEach(name => {{
        const inst = DATA.instances[name];
        const col = document.createElement('div');
        col.className = 'instance-col';

        const totalW = inst.sessions.reduce((a, s) => a + s.word_count, 0);
        const latest = inst.sessions.length > 0 ? inst.sessions[inst.sessions.length - 1] : null;

        col.innerHTML = `
            <div class="instance-header">
                <div class="instance-name ${{name}}">${{name}}</div>
                <div class="instance-stats">
                    ${{inst.sessions.length}} sessions \u00b7 ${{inst.evolutions.length}} evolutions \u00b7 ${{totalW.toLocaleString()}} words
                </div>
            </div>
            <div class="instance-latest">
                ${{latest ?
                    `<div class="instance-latest-label">Latest: #${{latest.num}} \u2014 ${{latest.date_display}}</div>
                     ${{latest.body || latest.first_line}}` :
                    '<em style="color:#3a3a4a">No sessions yet</em>'
                }}
            </div>
        `;
        colsDiv.appendChild(col);
    }});
    instTab.appendChild(colsDiv);

    // Theme comparison
    const themeTitle = document.createElement('h2');
    themeTitle.textContent = 'Theme comparison';
    themeTitle.style.cssText = 'font-size:1rem;font-weight:500;color:#8a8a9a;margin:1.5rem 0 1rem;padding-bottom:0.5rem;border-bottom:1px solid #1a1a2a';
    instTab.appendChild(themeTitle);

    // Find shared themes across instances
    const allInstThemes = {{}};
    instNames.forEach(name => {{
        const inst = DATA.instances[name];
        const themes = {{}};
        inst.sessions.forEach(s => s.themes.slice(0, 8).forEach(t => {{ themes[t] = (themes[t] || 0) + 1; }}));
        allInstThemes[name] = themes;
    }});
    const sharedThemes = new Set();
    if (instNames.length >= 2) {{
        const themeSets = instNames.map(n => new Set(Object.keys(allInstThemes[n])));
        themeSets[0].forEach(t => {{
            if (themeSets.slice(1).some(s => s.has(t))) sharedThemes.add(t);
        }});
    }}

    const compareDiv = document.createElement('div');
    compareDiv.className = 'theme-compare';

    instNames.forEach(name => {{
        const themes = allInstThemes[name];
        const sorted = Object.entries(themes).sort((a,b) => b[1]-a[1]).slice(0, 15);
        const col = document.createElement('div');
        col.className = 'theme-compare-col';
        col.innerHTML = `
            <h4 class="${{name}}">${{name}}</h4>
            <div>${{sorted.map(([w,c]) =>
                `<span class="theme-tag ${{c >= 3 ? 'hot' : ''}} ${{sharedThemes.has(w) ? 'shared' : ''}}">${{w}} <small>\u00d7${{c}}</small></span>`
            ).join('')}}</div>
        `;
        compareDiv.appendChild(col);
    }});
    instTab.appendChild(compareDiv);
}} else {{
    instTab.innerHTML = '<div style="padding:4rem;text-align:center;color:#3a3a4a;font-style:italic">No instances configured yet. Run setup_instances.py to create them.</div>';
}}

// ── Overview tab ────────────────────────────────────────────────
const overviewTab = document.getElementById('tab-overview');
overviewTab.innerHTML = `
    <div style="max-width:700px;margin-bottom:2.5rem">
        <h2 style="border:none;margin-top:0">About</h2>
        <p style="font-size:0.88rem;color:#9a9aaa;line-height:1.8;margin-bottom:1rem">
            The <strong style="color:#c8c8d0">Default Mode Network</strong> is modelled on the brain's default mode network \u2014 the neural circuitry that activates when you're not focused on anything in particular. Daydreaming. Shower thoughts. The place where unexpected connections form.
        </p>
        <p style="font-size:0.88rem;color:#9a9aaa;line-height:1.8;margin-bottom:1rem">
            This project lets Claude think without purpose, follow threads without destination, make connections without justification. Each session seeds from the last, creating a slow drift through idea-space over days and weeks.
        </p>
        <p style="font-size:0.88rem;color:#9a9aaa;line-height:1.8;margin-bottom:1rem">
            The system evolves itself \u2014 not toward a goal, but toward richer, stranger, more varied wandering. An evolution agent reviews every 5 sessions and modifies the system prompt, concept bank, and seeding rules. The evolution is the point, not the destination.
        </p>
        <div style="display:flex;gap:2rem;margin-top:1.5rem;padding:1rem;background:#0f0f18;border-radius:6px;border:1px solid #1a1a2a">
            <div style="flex:1">
                <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#5a5a6a;margin-bottom:0.3rem">Architecture</div>
                <div style="font-size:0.8rem;color:#8a8a9a;line-height:1.6">
                    <strong style="color:#6a9aba">dmn.py</strong> generates sessions using Claude Opus<br>
                    <strong style="color:#e08855">evolve.py</strong> reflects and modifies the system every 5 sessions<br>
                    <strong style="color:#8a8aaa">program.md</strong> is the evolvable DNA \u2014 prompt, concepts, rules
                </div>
            </div>
            <div style="flex:1">
                <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#5a5a6a;margin-bottom:0.3rem">Convergence Experiment</div>
                <div style="font-size:0.8rem;color:#8a8a9a;line-height:1.6">
                    Three control instances (<span style="color:#e74c3c">alpha</span>, <span style="color:#3498db">beta</span>, <span style="color:#2ecc71">gamma</span>) run from the same starting prompt with different seeds. Do they converge to the same themes? If so, the attractors are model-level, not prompt-level.
                </div>
            </div>
        </div>
        <div style="margin-top:1.5rem;padding:1rem;background:#0f0f18;border-radius:6px;border:1px solid #1a1a2a">
            <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#5a5a6a;margin-bottom:0.6rem">Bio-Inspired Experimental Instances</div>
            <p style="font-size:0.82rem;color:#7a7a8a;line-height:1.6;margin-bottom:0.8rem">
                Three additional instances test whether biological DMN mechanisms can break or reshape the attractor states found in the control group.
            </p>
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap">
                <div style="flex:1;min-width:180px">
                    <div style="font-size:0.78rem;margin-bottom:0.3rem"><span style="color:#f39c12">\u25cf</span> <strong style="color:#c8c8d0">replay</strong> \u2014 <em style="color:#7a7a8a">Selective Memory Replay</em></div>
                    <div style="font-size:0.75rem;color:#6a6a7a;line-height:1.5">
                        30% of sessions replay the most surprising past session instead of the most recent. Mimics hippocampal memory consolidation \u2014 the brain replays important memories during rest, not just the latest ones.
                    </div>
                </div>
                <div style="flex:1;min-width:180px">
                    <div style="font-size:0.78rem;margin-bottom:0.3rem"><span style="color:#9b59b6">\u25cf</span> <strong style="color:#c8c8d0">perturb</strong> \u2014 <em style="color:#7a7a8a">State Perturbation</em></div>
                    <div style="font-size:0.75rem;color:#6a6a7a;line-height:1.5">
                        Every 4th session injects an alien concept from a distant domain with a forced formal constraint. Mimics spontaneous state transitions \u2014 the biological DMN doesn\u2019t sit in one attractor but transitions between several.
                    </div>
                </div>
                <div style="flex:1;min-width:180px">
                    <div style="font-size:0.78rem;margin-bottom:0.3rem"><span style="color:#1abc9c">\u25cf</span> <strong style="color:#c8c8d0">switch</strong> \u2014 <em style="color:#7a7a8a">Task-Positive Switching</em></div>
                    <div style="font-size:0.75rem;color:#6a6a7a;line-height:1.5">
                        Alternates between wandering (odd sessions) and focused analytical critique (even sessions). Mimics DMN\u2013TPN anticorrelation \u2014 in the brain, task focus and mind-wandering suppress each other, and the switching prevents attractor lock-in.
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="stats">
        <div class="stat"><div class="stat-num">${{DATA.sessions.length}}</div><div class="stat-label">Sessions</div></div>
        <div class="stat"><div class="stat-num">${{DATA.evolutions.length}}</div><div class="stat-label">Evolutions</div></div>
        <div class="stat"><div class="stat-num">${{DATA.sessions.filter(s => s.concept).length}}</div><div class="stat-label">Concepts injected</div></div>
        <div class="stat"><div class="stat-num">${{totalWords.toLocaleString()}}</div><div class="stat-label">Total words</div></div>
    </div>

    <h2>Timeline</h2>
    <div class="timeline"><div class="timeline-track" id="overview-timeline"></div></div>

    <h2>Themes over time</h2>
    <div id="overview-themes"></div>

    <h2>Ending types</h2>
    <div class="legend">
        <div class="legend-item"><div class="legend-swatch ending-complete"></div> Complete</div>
        <div class="legend-item"><div class="legend-swatch ending-mid-sentence"></div> Mid-sentence</div>
        <div class="legend-item"><div class="legend-swatch ending-trailing"></div> Trailing</div>
    </div>
    <div class="form-grid" id="overview-endings"></div>

    <h2>Concept bank</h2>
    <div style="font-size:0.75rem;color:#4a4a5a;margin-bottom:0.5rem">
        <span style="color:#6a9aba">blue</span> = injected &nbsp;
        <span style="color:#3a3a4a">dim</span> = never used &nbsp;
        <span style="color:#8a4a4a;text-decoration:line-through">struck</span> = exhausted
    </div>
    <div class="concept-grid" id="overview-concepts"></div>
`;

// Timeline bars
const ovTimeline = document.getElementById('overview-timeline');
const maxW = Math.max(...DATA.sessions.map(s => s.word_count));
let allEvents = [];
DATA.sessions.forEach(s => allEvents.push({{type:'session', data:s}}));
DATA.evolutions.forEach(e => allEvents.push({{type:'evolution', data:e}}));
allEvents.sort((a, b) => (a.data.date || '').localeCompare(b.data.date || ''));
allEvents.forEach(evt => {{
    if (evt.type === 'session') {{
        const s = evt.data;
        const h = Math.max(4, (s.word_count / maxW) * 100);
        const bar = document.createElement('div');
        bar.className = 'session-bar' + (s.concept ? ' has-concept' : '');
        bar.style.height = h + 'px';
        bar.title = `#${{s.num}} — ${{s.word_count}} words`;
        bar.onclick = () => {{ switchTab('sessions'); showSession(s, null); }};
        ovTimeline.appendChild(bar);
    }} else {{
        const m = document.createElement('div');
        m.className = 'evolution-marker';
        ovTimeline.appendChild(m);
    }}
}});

// Theme timeline
const ovThemes = document.getElementById('overview-themes');
DATA.themeTimeline.forEach(w => {{
    const div = document.createElement('div');
    div.className = 'theme-window';
    div.innerHTML = `<div class="theme-window-label">Sessions ${{w.sessions}}</div>` +
        w.top_themes.map(t => `<span class="theme-tag ${{t.count >= 3 ? 'hot' : ''}}">${{t.word}} <small>\u00d7${{t.count}}</small></span>`).join('');
    ovThemes.appendChild(div);
}});

// Endings grid
const ovEndings = document.getElementById('overview-endings');
DATA.sessions.forEach(s => {{
    const cell = document.createElement('div');
    cell.className = `form-cell ending-${{s.ending_type}}`;
    cell.textContent = s.num;
    cell.title = `#${{s.num}} — ${{s.ending_type}}`;
    ovEndings.appendChild(cell);
}});

// Concept grid
const ovConcepts = document.getElementById('overview-concepts');
const usedConcepts = new Set(Object.keys(DATA.conceptUsage));
const exhaustedSet = new Set(DATA.exhaustedThemes.map(t => t.toLowerCase()));
const allConcepts = [...new Set([...DATA.conceptBank])];
allConcepts.sort((a, b) => {{
    const aU = usedConcepts.has(a), bU = usedConcepts.has(b);
    if (aU && !bU) return -1;
    if (!aU && bU) return 1;
    return a.localeCompare(b);
}});
allConcepts.forEach(c => {{
    const chip = document.createElement('span');
    const isEx = exhaustedSet.has(c.toLowerCase());
    const isU = usedConcepts.has(c);
    chip.className = `concept-chip ${{isEx ? 'concept-exhausted' : isU ? 'concept-used' : 'concept-unused'}}`;
    chip.textContent = c;
    if (isU) chip.title = `Sessions: ${{DATA.conceptUsage[c].join(', ')}}`;
    ovConcepts.appendChild(chip);
}});

// ── Analysis tab ────────────────────────────────────────────────
const analysis = DATA.analysis;
const INSTANCE_COLORS = {{
    main: '#e74c3c', alpha: '#3498db', beta: '#e67e22',
    gamma: '#2ecc71', replay: '#f39c12', perturb: '#9b59b6', switch: '#1abc9c', null: '#95a5a6'
}};
let convergenceChart = null;
let umapChart = null;
let activeInstances = new Set();

function buildFilterChips() {{
    const filterDiv = document.getElementById('analysis-filter');
    if (!analysis || !analysis.available) return;
    const r = analysis.results;
    const instances = Object.keys(r.session_counts || {{}});
    instances.forEach(name => {{
        const chip = document.createElement('label');
        chip.className = 'filter-chip active';
        chip.style.setProperty('--chip-color', INSTANCE_COLORS[name] || '#888');
        chip.innerHTML = `<input type="checkbox" checked value="${{name}}"><span class="filter-dot"></span>${{name}}`;
        chip.querySelector('input').addEventListener('change', function() {{
            if (this.checked) {{
                activeInstances.add(name);
                chip.classList.add('active');
            }} else {{
                activeInstances.delete(name);
                chip.classList.remove('active');
            }}
            updateUmapChart();
            updateConvergenceChart();
            updateVarianceChart();
        }});
        activeInstances.add(name);
        filterDiv.appendChild(chip);
    }});
}}

function updateUmapChart() {{
    if (!analysis || !analysis.available) return;
    const r = analysis.results;
    const umap = r.umap || {{}};

    const datasets = [];
    for (const [name, points] of Object.entries(umap)) {{
        if (!activeInstances.has(name)) continue;
        if (!points || points.length === 0) continue;

        const colour = INSTANCE_COLORS[name] || '#888';

        // Line dataset (trajectory)
        datasets.push({{
            label: name + ' trajectory',
            data: points.map(p => ({{ x: p[0], y: p[1] }})),
            borderColor: colour,
            borderWidth: 1,
            pointRadius: 0,
            showLine: true,
            fill: false,
            tension: 0,
            order: 2,
        }});

        // Points dataset
        datasets.push({{
            label: name,
            data: points.map(p => ({{ x: p[0], y: p[1] }})),
            backgroundColor: colour + 'bb',
            borderColor: colour,
            borderWidth: 0.5,
            pointRadius: 4,
            pointHoverRadius: 6,
            showLine: false,
            order: 1,
        }});

        // First point (larger, white border)
        datasets.push({{
            label: name + ' start',
            data: [{{ x: points[0][0], y: points[0][1] }}],
            backgroundColor: colour,
            borderColor: '#ffffff',
            borderWidth: 2,
            pointRadius: 8,
            showLine: false,
            order: 0,
        }});
    }}

    if (umapChart) umapChart.destroy();

    const ctx = document.getElementById('umap-canvas');
    if (!ctx) return;
    umapChart = new Chart(ctx, {{
        type: 'scatter',
        data: {{ datasets }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    labels: {{
                        color: '#ccc',
                        filter: (item) => !item.text.includes('trajectory') && !item.text.includes('start'),
                    }},
                }},
                title: {{
                    display: true,
                    text: 'DMN Attractor Map — Session Trajectories in Embedding Space',
                    color: '#fff',
                    font: {{ size: 14 }},
                }},
            }},
            scales: {{
                x: {{
                    title: {{ display: true, text: 'UMAP-1', color: '#666' }},
                    ticks: {{ color: '#888' }},
                    grid: {{ color: '#333' }},
                }},
                y: {{
                    title: {{ display: true, text: 'UMAP-2', color: '#666' }},
                    ticks: {{ color: '#888' }},
                    grid: {{ color: '#333' }},
                }},
            }},
        }},
    }});
}}

function updateConvergenceChart() {{
    if (!analysis || !analysis.available) return;
    const r = analysis.results;
    const convergence = r.convergence || {{}};

    // Filter to only pairs where both instances are active
    const datasets = [];
    const pairColors = [
        '#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6',
        '#1abc9c', '#f39c12', '#e84393', '#00cec9', '#fdcb6e',
        '#6c5ce7', '#a29bfe', '#ff7675', '#74b9ff', '#55efc4'
    ];
    let colorIdx = 0;

    for (const [pair, points] of Object.entries(convergence)) {{
        const parts = pair.split('\u2013');
        if (parts.length !== 2) continue;
        const [a, b] = parts;
        if (!activeInstances.has(a) || !activeInstances.has(b)) continue;
        if (points.length < 2) continue;

        // Use instance colors blended
        const color = pairColors[colorIdx % pairColors.length];
        colorIdx++;
        datasets.push({{
            label: pair,
            data: points.map(p => ({{ x: p[0], y: p[1] }})),
            borderColor: color,
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
        }});
    }}

    if (convergenceChart) {{
        convergenceChart.data.datasets = datasets;
        convergenceChart.update();
    }} else {{
        const ctx = document.getElementById('convergence-canvas');
        if (!ctx) return;
        convergenceChart = new Chart(ctx, {{
            type: 'line',
            data: {{ datasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{ color: '#8888aa', font: {{ size: 11 }}, boxWidth: 20 }}
                    }}
                }},
                scales: {{
                    x: {{
                        type: 'linear',
                        title: {{ display: true, text: 'Session index', color: '#6666aa' }},
                        grid: {{ color: 'rgba(100,100,150,0.15)' }},
                        ticks: {{ color: '#6666aa' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Cosine distance (lower = more similar)', color: '#6666aa' }},
                        grid: {{ color: 'rgba(100,100,150,0.15)' }},
                        ticks: {{ color: '#6666aa' }},
                        min: 0, max: 1
                    }}
                }}
            }}
        }});
    }}
}}

let varianceChart = null;
function updateVarianceChart() {{
    if (!analysis || !analysis.available) return;
    const r = analysis.results;
    const variance = r.variance || {{}};

    const datasets = [];
    for (const [name, points] of Object.entries(variance)) {{
        if (!activeInstances.has(name)) continue;
        if (points.length < 2) continue;
        datasets.push({{
            label: name,
            data: points.map(p => ({{ x: p[0], y: p[1] }})),
            borderColor: INSTANCE_COLORS[name] || '#888',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
        }});
    }}

    if (varianceChart) {{
        varianceChart.data.datasets = datasets;
        varianceChart.update();
    }} else {{
        const ctx = document.getElementById('variance-canvas');
        if (!ctx) return;
        varianceChart = new Chart(ctx, {{
            type: 'line',
            data: {{ datasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{ color: '#8888aa', font: {{ size: 11 }}, boxWidth: 20 }}
                    }}
                }},
                scales: {{
                    x: {{
                        type: 'linear',
                        title: {{ display: true, text: 'Session index', color: '#6666aa' }},
                        grid: {{ color: 'rgba(100,100,150,0.15)' }},
                        ticks: {{ color: '#6666aa' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Mean embedding variance (5-session window)', color: '#6666aa' }},
                        grid: {{ color: 'rgba(100,100,150,0.15)' }},
                        ticks: {{ color: '#6666aa' }}
                    }}
                }}
            }}
        }});
    }}
}}

if (analysis && analysis.available) {{
    const summary = document.getElementById('analysis-summary');
    const r = analysis.results;
    if (r.summary) {{
        const trend = r.summary.trend;
        const trendClass = trend === 'converging' ? 'converging' : 'diverging';
        const trendSymbol = trend === 'converging' ? '↓' : '↑';
        const ratio = r.summary.convergence_ratio ? r.summary.convergence_ratio.toFixed(2) : '—';
        summary.innerHTML = `
            <div class="analysis-stat">
                <div class="analysis-stat-val">${{Object.values(r.session_counts).reduce((a,b)=>a+b,0)}}</div>
                <div class="analysis-stat-label">Total sessions analysed</div>
            </div>
            <div class="analysis-stat">
                <div class="analysis-stat-val">${{Object.keys(r.session_counts).length}}</div>
                <div class="analysis-stat-label">Instances</div>
            </div>
            <div class="analysis-stat">
                <div class="analysis-stat-val">${{r.summary.early_mean_distance.toFixed(3)}}</div>
                <div class="analysis-stat-label">Early distance (sessions 0–9)</div>
            </div>
            <div class="analysis-stat">
                <div class="analysis-stat-val">${{r.summary.late_mean_distance.toFixed(3)}}</div>
                <div class="analysis-stat-label">Late distance (last 10)</div>
            </div>
            <div class="analysis-stat">
                <div class="analysis-stat-val ${{trendClass}}">${{trendSymbol}} ${{trend.toUpperCase()}}</div>
                <div class="analysis-stat-label">Overall trend (ratio: ${{ratio}})</div>
            </div>
        `;
    }}

    // Insert static images (UMAP, distance, entropy)
    const imgMap = {{
        'umap_attractor_map': 'img-umap',
    }};
    for (const [key, elemId] of Object.entries(imgMap)) {{
        const wrap = document.getElementById(elemId);
        if (!wrap) continue;
        if (analysis.images[key]) {{
            wrap.innerHTML = `<img src="${{analysis.images[key]}}" alt="${{key}}">`;
        }} else {{
            wrap.innerHTML = `<div class="analysis-empty">Plot not available — run python3 analyse.py</div>`;
        }}
    }}

    // Build filter chips and charts
    buildFilterChips();
    // Charts need visible container — defer if analysis tab not active
    function initCharts() {{
        updateUmapChart();
        updateConvergenceChart();
        updateVarianceChart();
    }}
    if (document.getElementById('tab-analysis').classList.contains('active')) {{
        initCharts();
    }} else {{
        // Patch switchTab to init charts on first visit
        const _origSwitch = window.switchTab;
        window.switchTab = function(name) {{
            _origSwitch(name);
            if (name === 'analysis' && !convergenceChart) {{
                setTimeout(initCharts, 50);
            }}
        }};
    }}
}} else {{
    document.getElementById('tab-analysis').innerHTML = `
        <div class="analysis-empty" style="padding: 6rem 2rem;">
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">No analysis data yet</p>
            <p>Run <code style="background:#1a1a2a;padding:0.2rem 0.5rem;border-radius:3px;">python3 analyse.py</code> to generate embeddings and attractor visualisations.</p>
        </div>
    `;
}}

</script>
</body>
</html>"""

    return html


def gather_instance_data():
    """Gather data for all instances in instances/ directory."""
    instances_dir = DIR / "instances"
    if not instances_dir.exists():
        return {}

    instances = {}
    for inst_dir in sorted(instances_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        name = inst_dir.name
        sessions_dir = inst_dir / "sessions"
        evolutions_dir = inst_dir / "evolutions"
        if not sessions_dir.exists():
            continue

        inst_sessions = parse_sessions(sessions_dir)
        inst_evolutions = parse_evolutions(evolutions_dir) if evolutions_dir.exists() else []
        inst_concepts, inst_exhausted = parse_concept_bank(inst_dir / "program.md")
        inst_usage = build_concept_usage(inst_sessions)
        inst_themes = build_theme_timeline(inst_sessions)

        inst_state = {}
        state_file = inst_dir / ".dmn_state.json"
        if state_file.exists():
            inst_state = json.loads(state_file.read_text())

        instances[name] = {
            "sessions": inst_sessions,
            "evolutions": inst_evolutions,
            "conceptBank": inst_concepts,
            "exhaustedThemes": inst_exhausted,
            "conceptUsage": inst_usage,
            "themeTimeline": inst_themes,
            "state": inst_state,
            "sessionCount": len(inst_sessions),
            "evolutionCount": len(inst_evolutions),
        }

    return instances


def main():
    sessions = parse_sessions()
    evolutions = parse_evolutions()
    concept_bank, exhausted = parse_concept_bank()
    concept_usage = build_concept_usage(sessions)
    theme_timeline = build_theme_timeline(sessions)
    reflections = parse_reflections()

    state = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())

    # Gather instance data
    instances = gather_instance_data()

    html = generate_html(
        sessions, evolutions, concept_bank, exhausted,
        concept_usage, theme_timeline, reflections, state,
        instances=instances
    )

    OUTPUT_FILE.write_text(html)
    print(f"Dashboard generated: {OUTPUT_FILE}")
    print(f"  {len(sessions)} sessions, {len(evolutions)} evolutions")
    print(f"  {len(concept_bank)} concepts in bank, {len(concept_usage)} injected")
    if instances:
        for name, data in instances.items():
            print(f"  [{name}] {data['sessionCount']} sessions, {data['evolutionCount']} evolutions")


if __name__ == "__main__":
    main()
