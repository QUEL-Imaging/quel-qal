"""Look up product presets from a catalog SKU, unit serial, or fragment.

Usage:
    python -m qal.examples.general.target_preset_lookup_example
    python -m qal.examples.general.target_preset_lookup_example RCS-ICG-ST
    python -m qal.examples.general.target_preset_lookup_example RCS-ICG --all
"""

from __future__ import annotations

from argparse import ArgumentParser

from qal.util import find_target_presets, lookup_target_preset, target_data_labels


# Catalog SKU, unit serial, unique fragment, or family-scoped tag.
# Examples: "RCS-ICG-ST01-QUEL03", "RCS-S-IC1-S1-A-113", "RDS-80Q-LU05".
TARGET_QUERY = "V3-ICG-ST03-RDS007"


def _print_preset(preset: dict) -> None:
    labels = target_data_labels(preset)
    print(preset.get("display_name") or preset.get("tag"))
    print(
        f"  family: {preset.get('family_key')} ({preset.get('product_code')})"
    )
    print(f"  tag: {preset.get('tag')}")
    for field in ("old_sku", "current_sku"):
        value = preset.get(field)
        if value:
            print(f"  {field}: {value}")
    well_ids = labels.get("well_ids")
    if well_ids:
        print("  well_ids:")
        for well_id in well_ids:
            print(f"    - {well_id}")
    for field in (
        "fluorophore_label",
        "qal_graph_type",
        "chart",
        "groups",
        "channel_nm",
        "peak_wavelength_nm",
        "well_diameter_mm",
    ):
        if field in labels:
            print(f"  {field}: {labels[field]}")


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
        matches = find_target_presets(args.query)
        if not matches:
            raise SystemExit(f"No product preset matches {args.query!r}")
        for preset in matches:
            _print_preset(preset)
            print()
        return

    _print_preset(lookup_target_preset(args.query))


if __name__ == "__main__":
    main()
