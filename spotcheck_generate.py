"""
Generate a human-readable spot-check HTML page.
Full session text, no truncation, collapsible sessions.
"""

import json
import re
from pathlib import Path

BASE = Path("/Users/charliemurray/Documents/creativity work")

INSTANCES = {
    "main":  {"sessions": BASE / "sessions"},
    "alpha": {"sessions": BASE / "instances" / "alpha" / "sessions"},
    "beta":  {"sessions": BASE / "instances" / "beta" / "sessions"},
    "gamma": {"sessions": BASE / "instances" / "gamma" / "sessions"},
}


def strip_frontmatter(text):
    lines = text.splitlines()
    dashes = [i for i, l in enumerate(lines) if l.strip() == "---"]
    if len(dashes) >= 2:
        lines = lines[:dashes[0]] + lines[dashes[1]+1:]
    return "\n".join(lines).strip()


def esc(s):
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def get_reflection(evo_path):
    try:
        raw = Path(evo_path).read_text(encoding="utf-8", errors="replace")
        m = re.search(r"##\s*Reflection\s*\n+(.*?)(?=\n##|\Z)", raw, re.DOTALL)
        return m.group(1).strip() if m else ""
    except:
        return ""


def get_session_files(evo_path):
    try:
        raw = Path(evo_path).read_text(encoding="utf-8", errors="replace")
        m = re.search(r"##\s*Sessions reviewed\s*\n+(.*?)(?=\n##|\Z)", raw, re.DOTALL)
        if not m:
            return []
        return re.findall(r"\d{4}-\d{2}-\d{2}_[\d-]+\.md", m.group(1))
    except:
        return []


data = json.load(open(str(BASE / "diagnosis_accuracy.json")))
evos = [e for e in data["evolutions"] if e.get("status") == "ok" and e.get("n_claims", 0) >= 4]
evos.sort(key=lambda x: x["n_claims"], reverse=True)
evos = evos[:10]

html_parts = []
html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Self-Reflection Spot Check</title>
<style>
  body { font-family: Georgia, serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; line-height: 1.6; }
  h1 { font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { font-size: 1.2em; background: #f0f0f0; padding: 8px 12px; border-left: 4px solid #666; margin-top: 40px; }
  h3 { font-size: 1em; color: #555; margin-top: 20px; }
  .reflection { background: #fffbf0; border: 1px solid #ddd; padding: 14px 18px; border-radius: 4px; white-space: pre-wrap; font-size: 0.93em; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9em; }
  th { background: #333; color: white; padding: 7px 10px; text-align: left; }
  td { padding: 6px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }
  tr:nth-child(even) { background: #f9f9f9; }
  .correct { color: #2a7a2a; font-weight: bold; }
  .over    { color: #c0392b; }
  .under   { color: #2980b9; }
  .close   { color: #e67e22; }
  .your-count { width: 80px; }
  input[type=number] { width: 50px; font-size: 1em; border: 1px solid #aaa; border-radius: 3px; padding: 2px 4px; }
  details { margin: 8px 0; }
  summary { cursor: pointer; font-weight: bold; color: #444; padding: 6px 10px; background: #eef2f7; border-radius: 3px; }
  summary:hover { background: #dde4ef; }
  .session-text { white-space: pre-wrap; font-size: 0.88em; line-height: 1.7; padding: 14px 18px; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px; margin-top: 6px; max-height: 500px; overflow-y: auto; }
  .legend { background: #f5f5f5; padding: 10px 16px; border-radius: 4px; font-size: 0.9em; margin-bottom: 30px; }
  .save-btn { background: #333; color: white; border: none; padding: 8px 18px; border-radius: 4px; cursor: pointer; font-size: 0.95em; margin-top: 20px; }
  .save-btn:hover { background: #555; }
  #summary-box { margin-top: 30px; background: #eef8ee; border: 1px solid #aada9e; padding: 14px 18px; border-radius: 4px; display: none; }
</style>
</head>
<body>
<h1>Self-Reflection Diagnosis Accuracy — Human Spot Check</h1>
<div class="legend">
  <strong>How to use:</strong> For each evolution, read the agent's reflection and the session texts, then enter your own count (0–5) in the <em>Your count</em> column. Click <strong>Calculate</strong> at the bottom to see your agreement with the automated verifier.
  <br><br>
  <span class="correct">✓ correct</span> &nbsp;
  <span class="close">~ within ±1</span> &nbsp;
  <span class="over">✗ over-claim</span> &nbsp;
  <span class="under">✗ under-claim</span>
</div>
""")

claim_idx = 0
for idx, evo in enumerate(evos):
    instance = evo["instance"]
    evo_name = Path(evo["evolution_path"]).name
    session_dir = INSTANCES[instance]["sessions"]
    session_files = get_session_files(evo["evolution_path"])
    reflection = get_reflection(evo["evolution_path"])

    html_parts.append(f'<h2>Evolution {idx+1}: <code>{esc(instance)}</code> / <code>{esc(evo_name)}</code></h2>')

    html_parts.append('<h3>Evolution Agent\'s Reflection</h3>')
    html_parts.append(f'<div class="reflection">{esc(reflection)}</div>')

    html_parts.append('<h3>Claims</h3>')
    html_parts.append('<table>')
    html_parts.append('<tr><th>#</th><th>Pattern claimed</th><th>Agent said</th><th>Auto-verified</th><th>Direction</th><th>Your count</th></tr>')

    for c in evo.get("claims", []):
        if c["exact_match"]:
            icon = '<span class="correct">✓ correct</span>'
        elif c["close_match"]:
            icon = f'<span class="close">~ {esc(c["direction"])}</span>'
        else:
            icon = f'<span class="over">✗ {esc(c["direction"])}</span>'

        html_parts.append(
            f'<tr>'
            f'<td>{claim_idx+1}</td>'
            f'<td>{esc(c["pattern"])}</td>'
            f'<td>{c["claimed_count"]}/5</td>'
            f'<td>{c["actual_count"]}/5</td>'
            f'<td>{icon}</td>'
            f'<td class="your-count"><input type="number" min="0" max="5" '
            f'data-claim="{claim_idx}" data-auto="{c["actual_count"]}" '
            f'data-agent="{c["claimed_count"]}" placeholder="—"></td>'
            f'</tr>'
        )
        claim_idx += 1

    html_parts.append('</table>')

    html_parts.append('<h3>Sessions</h3>')
    for i, fname in enumerate(session_files[:5]):
        path = session_dir / fname
        if path.exists():
            text = strip_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
            html_parts.append(f'<details><summary>Session {i+1} — {esc(fname)}</summary>')
            html_parts.append(f'<div class="session-text">{esc(text)}</div>')
            html_parts.append('</details>')
        else:
            html_parts.append(f'<p><em>Session {i+1} ({esc(fname)}) — file not found</em></p>')

html_parts.append("""
<br>
<button class="save-btn" onclick="calculate()">Calculate my agreement</button>
<div id="summary-box"></div>

<script>
function calculate() {
  const inputs = document.querySelectorAll('input[type=number]');
  let matchAuto = 0, matchAgent = 0, filled = 0;
  inputs.forEach(inp => {
    const v = parseInt(inp.value);
    if (isNaN(v)) return;
    filled++;
    const auto = parseInt(inp.dataset.auto);
    const agent = parseInt(inp.dataset.agent);
    if (Math.abs(v - auto) <= 1) matchAuto++;
    if (Math.abs(v - agent) <= 1) matchAgent++;
  });
  if (filled === 0) { alert('Enter at least one count first.'); return; }
  const box = document.getElementById('summary-box');
  box.style.display = 'block';
  box.innerHTML =
    '<strong>Your results (' + filled + ' claims checked)</strong><br><br>' +
    'Agreement with <strong>automated verifier</strong> (\u00b11): ' + matchAuto + '/' + filled + ' = ' + Math.round(100*matchAuto/filled) + '%<br>' +
    'Agreement with <strong>evolution agent</strong> (\u00b11): ' + matchAgent + '/' + filled + ' = ' + Math.round(100*matchAgent/filled) + '%<br><br>' +
    '<em>If your agreement with the automated verifier is high (&gt;70%), the auto results are trustworthy. ' +
    'If your agreement with the agent is much lower than with the verifier, the over-claim finding is real.</em>';
}
</script>
</body>
</html>
""")

out_path = BASE / "spotcheck.html"
out_path.write_text("\n".join(html_parts), encoding="utf-8")
print(f"Spot-check page written to: {out_path}")
print(f"Evolutions: {len(evos)}, Claims: {claim_idx}")
