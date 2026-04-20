#!/usr/bin/env python3
"""
Set up 3 parallel DMN instances for the convergence experiment.

Each instance starts from the same program.md but with a different initial seed:
  - alpha: no concept (pure drift from blank)
  - beta: seeded with "hydraulics" (mechanical/physical)
  - gamma: seeded with "lullaby" (musical/emotional)

They evolve independently. If they converge to the same themes,
the attractors are model-level. If they diverge, they're path-dependent.
"""

import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
INSTANCES_DIR = BASE_DIR / "instances"
SOURCE_PROGRAM = BASE_DIR / "program.md"
SOURCE_CLAUDE = BASE_DIR / "CLAUDE.md"

INSTANCES = {
    "alpha": {
        "description": "Pure drift — no initial concept seed",
        "initial_concept": None,
    },
    "beta": {
        "description": "Mechanical/physical seed — hydraulics",
        "initial_concept": "hydraulics",
    },
    "gamma": {
        "description": "Musical/emotional seed — lullaby",
        "initial_concept": "lullaby",
    },
}


def setup():
    INSTANCES_DIR.mkdir(exist_ok=True)

    for name, config in INSTANCES.items():
        inst_dir = INSTANCES_DIR / name
        if inst_dir.exists():
            print(f"  [{name}] Already exists — skipping")
            continue

        inst_dir.mkdir()
        (inst_dir / "sessions").mkdir()
        (inst_dir / "evolutions").mkdir()

        # Copy program.md (identical starting point)
        shutil.copy2(SOURCE_PROGRAM, inst_dir / "program.md")

        # Create a minimal CLAUDE.md for each instance
        claude_md = f"""# DMN — Instance: {name}

{config['description']}

## Reflections

*No reflections yet. This section will grow as the instance runs and evolves.*
"""
        (inst_dir / "CLAUDE.md").write_text(claude_md)

        # Initialise state with different seeds
        state = {
            "session_count": 0,
            "last_excerpt": None,
            "last_concepts": [config["initial_concept"]] if config["initial_concept"] else [],
        }
        with open(inst_dir / ".dmn_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"  [{name}] Created — {config['description']}")

    print(f"\nDone. 3 instances ready in {INSTANCES_DIR}/")
    print("Run with: python dmn.py --instance alpha")


if __name__ == "__main__":
    setup()
