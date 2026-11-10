"""Look up product presets from a catalog SKU, unit serial, or fragment.

Usage:
    python -m qal.examples.general.target_preset_lookup_example
    python -m qal.examples.general.target_preset_lookup_example RCS-ICG-ST
    python -m qal.examples.general.target_preset_lookup_example RCS-ICG --all

The matching notebook is
``qal/examples/jupyter-notebook/target_preset_lookup_example.ipynb``.
"""

from __future__ import annotations

from argparse import ArgumentParser

from qal.util import find_target_presets, lookup_target_preset, print_preset


# Catalog SKU, unit serial, unique fragment, or family-scoped tag.
# Examples: "RCS-ICG-ST01-QUEL03", "RCS-S-IC1-S1-A-113", "RDS-80Q-LU05".
TARGET_QUERY = "V3-ICG-ST03-RDS007"


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
    args = parser.parse_args()

    if args.all:
        try:
            matches = find_target_presets(args.query)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if not matches:
            raise SystemExit(f"No product preset matches {args.query!r}")
        for preset in matches:
            print_preset(preset)
            print()
        return

    try:
        preset = lookup_target_preset(args.query)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print_preset(preset)


if __name__ == "__main__":
    main()
