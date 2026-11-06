# QUEL product presets

These YAML files contain product-family metadata used by or relevant to QAL
workflows:

- `concentration.yaml`
- `depth.yaml`
- `resolution.yaml`
- `uniformity_distortion.yaml`
- `radiometric_emitter.yaml`
- `depth_resolution.yaml`
- `reference_sets.yaml`

Catalog SKUs follow `FAMILY-FLUOROPHORE-FORM-QUEL`, for example
`RCS-ICG-ST01-QUEL03`. Family codes are RCS (concentration), RDS (depth), and
RRT (resolution). Current fluorophore codes are ICG, 80Q, and 70Q. Lookup also
accepts older serials for backwards compatibility:

- ICG depth ST01/ST02 serials map to current ST03 (`RDS-ICG-ST03-QUEL03`),
  which supersedes `RDS-ICG-ST02-QUEL02`
- OTL38 (`O38` / `038`) RCS/RDS/RRT units map to superseded O38 presets
  (1–1000 nM), not current Q800 (1–600 nM)
- Methylene blue (`MBL`) inventory units use the same 3x3 layouts
- Older shorthand serials such as `v2_R3_C07` use C/D/R for
  concentration/depth/resolution and default to standard ICG
- Custom RET serials such as `630-PW01-RRL001` synthesize a preset. The
  wavelength token fills `peak_wavelength_nm` and `serial_template`; catalog
  SKUs and fluorophore labels stay unset. Packaged 720 / 800 / 825 units still
  win, including q800's explicit 805 nm peak.

Look up a SKU, a unit serial, or a unique fragment with
`qal.util.lookup_target_preset`, or run
`python -m qal.examples.general.target_preset_lookup_example`.

Each product record includes identifier aliases used in the field:

- `old_sku`: catalog name printed on product records
- `current_sku`: internal SKU from the master list
- `serial_template`: current unit serial pattern (`xxx` is the unit number)
- `legacy_serial_template`: previous unit serial pattern
- `superseded_skus`: older catalog names that map to the current product

A null `old_sku` remains only when the catalog has no assigned identifier, such
as the depth-resolution V-wedge. Image-specific acquisition metadata, such as
the YAML sidecars under `qal/data/RET_images`, remains with the corresponding
images and is not product catalog data.
