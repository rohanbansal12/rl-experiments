from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE_SPEC = """# {name} SPEC\n\n## Goal\n\nTODO.\n\n## User-owned TODOs\n\n- TODO(user): algorithmic core\n\n## Agent-safe TODOs\n\n- configs\n- logging\n- tests\n- Modal launch support\n"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Project directory name, e.g. p03_dqn_rainbow")
    args = parser.parse_args()

    root = Path("rl_lab/projects") / args.name
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    (root / "tests").mkdir()
    (root / "__init__.py").write_text(f'"""{args.name}."""\n', encoding="utf-8")
    (root / "SPEC.md").write_text(TEMPLATE_SPEC.format(name=args.name), encoding="utf-8")
    (root / "README.md").write_text(f"# {args.name}\n\nTODO.\n", encoding="utf-8")
    print(f"Created {root}")


if __name__ == "__main__":
    main()
