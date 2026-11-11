# Recursive example consistency checklist

Scope: every `.py` and `.ipynb` file under `qal/examples/`, excluding the
`uniformity_distortion_rud/` directory. The RUD notebook is outside that
directory and is included. Checked items mean implementation was reviewed and
the applicable automated checks passed; they do not claim a manual GUI session.

For each runnable image example: `SAVE_RESULTS = True`,
`SAVE_SOURCE_IMAGES = False` with an implemented opt-in export, no copying of
user-supplied images, and a defined output folder outside the sample cache.
Lookup-only examples have no source image; source-image export is not applicable.
Package initializers and the output helper are reviewed separately.

Inventory: 19 runnable scripts/notebooks and 9 support files.

- [x] `__init__.py` — package initializer; no analysis or export settings apply.
- [x] `_output.py` — Reviewed shared paths, original-file copying, multiple-image grouping, and YAML exports; exercised by the script/notebook export tests.
- [x] `concentration_rcs/__init__.py` — package initializer; no analysis or export settings apply.
- [x] `concentration_rcs/concentration_automatic_example.py` — Both save controls, ROI/curve/statistics and target YAML tested with built-in samples, user files and saving disabled.
- [x] `concentration_rcs/concentration_gui_example.py` — Both save controls, ROI/curve/statistics and target YAML tested with built-in samples, user files and saving disabled.
- [x] `concentration_rcs/concentration_sku_gui_example.py` — Both save controls, ROI/curve/statistics and target YAML tested with built-in samples, user files and saving disabled. Serial query and resolved preset are retained in target.yaml.
- [x] `depth_rds/__init__.py` — package initializer; no analysis or export settings apply.
- [x] `depth_rds/depth_automatic_example.py` — Both save controls, ROI/curve/statistics and target YAML tested with built-in samples, user files and saving disabled.
- [x] `depth_rds/depth_gui_example.py` — Both save controls, ROI/curve/statistics and target YAML tested with built-in samples, user files and saving disabled.
- [x] `depth_rds/depth_sku_gui_example.py` — Both save controls, ROI/curve/statistics and target YAML tested with built-in samples, user files and saving disabled. Serial query and resolved preset are retained in target.yaml.
- [x] `depth_resolution_dr/__init__.py` — package initializer; no analysis or export settings apply.
- [x] `depth_resolution_dr/depth_resolution_example.py` — Both save controls and all four profile plots tested; one sample per run, backend helper and Enter/close confirmation reviewed.
- [x] `general/__init__.py` — package initializer; no analysis or export settings apply.
- [x] `general/target_preset_lookup_example.py` — Default YAML saving, query/matches/partial diagnostics, ICG fragment, and error handling checked; source-image saving is not applicable.
- [x] `jupyter-notebook/concentration_rcs_gui_example.ipynb` — Both save controls, ROI/curve/statistics and target YAML tested with built-in samples, user files and saving disabled.
- [x] `jupyter-notebook/depth_rds_gui_example.ipynb` — Both save controls, ROI/curve/statistics and target YAML tested with built-in samples, user files and saving disabled.
- [x] `jupyter-notebook/radiometric_emitter_ret_gui_example.ipynb` — Both save controls and original light/dark exports tested; optional dark frame and single stats.csv retained; no duplicate wells.csv.
- [x] `jupyter-notebook/resolution_rrt_gui_example.ipynb` — Both save controls, contrast CSV/plots and separate roi.png checked; original source-coordinate overlay saves without another window.
- [x] `jupyter-notebook/target_preset_lookup_example.ipynb` — Default YAML saving, query/matches/partial diagnostics, ICG fragment, and error handling checked; source-image saving is not applicable.
- [x] `jupyter-notebook/uniformity_distortion_rud_gui_example.ipynb` — Notebook outside the excluded folder: both controls, grouped sources and all CSV/figure exports tested, including figures closed by display.
- [x] `lung_phantom/__init__.py` — package initializer; no analysis or export settings apply.
- [x] `lung_phantom/create_lung_reference.py` — Added both save controls; real reference image/mask/metadata exports and two-source grouping tested for samples, user files, and saving disabled.
- [x] `lung_phantom/lung_phantom_example.py` — Added both save controls; real Excel metric exports and source-image handling tested; retained registration stopping-condition explanation.
- [x] `radiometric_emitter_ret/__init__.py` — package initializer; no analysis or export settings apply.
- [x] `radiometric_emitter_ret/ret_gui_example.py` — Both save controls and original light/dark exports tested; optional dark frame and single stats.csv retained; no duplicate wells.csv.
- [x] `radiometric_emitter_ret/ret_light_frame_automatic_analysis_example.py` — Both save controls and original light/dark exports tested; optional dark frame and single stats.csv retained; no duplicate wells.csv.
- [x] `resolution_rrt/__init__.py` — package initializer; no analysis or export settings apply.
- [x] `resolution_rrt/resolution_gui_example.py` — Both save controls, contrast CSV/plots and separate roi.png checked; original source-coordinate overlay saves without another window.

## Original review requirements

- [x] Concentration ROI alignment and built-in sample saving. Well ordering/ROI regressions and real-sample saved-output checks pass.
- [x] RCS/RDS SKU examples and query/preset YAML. Both SKU runners export the serial query and complete preset; all six grid runners pass the three input/saving scenarios.
- [x] Depth-resolution platform-aware backend, Enter confirmation, one sample per run. Backend helper and both confirmation views reviewed; profile exports pass with controlled crop bounds.
- [x] Partial malformed identifiers and an `ICG` fragment example. Malformed-serial regression tests, CLI lookup exports and both notebook lookup paths pass.
- [x] RET GUI consolidation, optional dark frame, automatic ROI display, no duplicate wells.csv. Both RET runners and the notebook pass export checks; raw/dark-pair and cancellation regressions pass.
- [x] RRT saves a standalone source-coordinate ROI overlay without another window. Script and notebook export roi.png; the unmanaged-figure regression verifies source coordinates without a pyplot window.
- [x] Lung registration stopping-condition explanation. Comments reviewed against the optimizer call; both real bundled-sample runs pass, including reference generation and Excel metrics.
- [x] RET diameter tool remains a documented follow-up, as requested. Documented in the RET guide; no diameter-tool implementation was added.

## Verification

- Recursive audit: **81 passed**, covering all 28 inventoried files. Image
  runners and notebooks were exercised with built-in sample copying, user images,
  and `SAVE_RESULTS = False`. Real numerical analysis and file writers were used;
  deterministic selection/detection fixtures avoided manual GUI input.
- Full suite on Python 3.13: **527 passed, 8 skipped, 1 deselected** using
  `MPLBACKEND=Agg python3.13 -m pytest -q -k 'not batch'`. The deselection is the
  repository's existing deferred batch-example policy; optional data tests skip
  unless enabled.
- Both lung examples also passed with the actual bundled images after hydrating
  their Git LFS files. Registration converged in 77 iterations / 152 function
  evaluations (about 27 seconds on this machine). Both Excel tables contained
  five inclusions; reference files and original sample copies were verified.
- The excluded `uniformity_distortion_rud/` directory has no changes.
- Desktop mouse interaction remains a manual smoke check; Enter confirmation
  and saved figure behavior are covered by headless regression tests.
