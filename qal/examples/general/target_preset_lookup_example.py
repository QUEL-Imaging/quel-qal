"""Look up product presets from a catalog SKU, unit serial, or fragment.

Usage:
    python -m qal.examples.general.target_preset_lookup_example
    python -m qal.examples.general.target_preset_lookup_example RCS-ICG-ST
    python -m qal.examples.general.target_preset_lookup_example RCS-ICG --all
    python -m qal.examples.general.target_preset_lookup_example ICG --all

The matching notebook is
``qal/examples/jupyter-notebook/target_preset_lookup_example.ipynb``.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import yaml

from qal.util import describe_target_query, find_target_presets, lookup_target_preset, print_preset


# Catalog SKU, unit serial, unique fragment, or family-scoped tag.
# Examples: "RCS-ICG-ST01-QUEL03", "RCS-S-IC1-S1-A-113", "RDS-80Q-LU05".
TARGET_QUERY = "V3-ICG-ST03-RDS007"
SAVE_RESULTS = True
OUTPUT_DIR = Path(__file__).resolve().parent / "target_preset_lookup_example_results"


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="?",
        default=TARGET_QUERY,
        help="SKU, serial, or fragment; defaults to TARGET_QUERY",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print every match instead of requiring a unique preset",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write lookup YAML")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    def save(matches):
        if SAVE_RESULTS and not args.no_save:
            args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
            path = args.output_dir.expanduser() / "target_lookup.yaml"
            path.write_text(yaml.safe_dump({
                "query": args.query,
                "diagnostic": describe_target_query(args.query),
                "matches": matches,
            }, sort_keys=False), encoding="utf-8")

    if args.all:
        try:
            matches = find_target_presets(args.query)
        except ValueError as exc:
            save([])
            raise SystemExit(str(exc)) from None
        diagnostic = describe_target_query(args.query)
        if diagnostic:
            print(diagnostic)
        if not matches:
            save([])
            raise SystemExit(f"No product preset matches {args.query!r}")
        save(matches)
        for preset in matches:
            print_preset(preset)
            print()
        return

    try:
        preset = lookup_target_preset(args.query)
    except ValueError as exc:
        save([])
        raise SystemExit(str(exc)) from None
    print_preset(preset)
    save([preset])


if __name__ == "__main__":
    main()
