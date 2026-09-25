# Well Detector GUI Guide

The well-selection GUI supports two coordinate modes:

- `"detect"` refines each click with a local threshold/flood-fill search around
  that coordinate and replaces it with the resulting centroid and ROI diameter.
- `"manual"` treats clicks as authoritative coordinates and does not run an
  additional image search.

The selected points can be used for a single RET well or as row-major anchors
for a 3×3 concentration/depth target.

## Controls

- **Left-click** inside the image to register a point.
- In manual-coordinate mode, **double left-click** a background region to add
  one optional Control. It is shown as a green `C`.
- **Right-click** close to a `+` marker to remove that point.
- **Delete/Backspace** removes the most recently added point.
- With live preview enabled, **left-click and drag a `+`** to reposition a
  point. A 3×3 overlay updates for concentration/depth anchors; a single RET
  point shows that well's ROI circle when `manual_roi_diameter` is set.
- **Enter/Return** confirms the current points.
- **Zoom/Pan** toolbar modes temporarily disable point addition and removal, so
  mouse actions used to navigate the image do not change the selection.
- The table on the right updates after every addition or removal.
- In detection mode, an animated status indicator remains visible while well
  detection and centroid refinement run in the background.
- `x` increases from left to right and `y` increases from top to bottom.

The markers use colors from Matplotlib's `plasma` colormap, matching the
detected-well overlays. A thin black outline keeps each marker visible against
both bright and dark grayscale regions.

![RET GUI with one selected coordinate](gui_images/ret_gui_selection.png)

## RET example

From the repository root:

```bash
python -m qal.examples.radiometric_emitter_ret.ret_gui_example
```

From `qal/examples/radiometric_emitter_ret`:

```bash
python ret_gui_example.py
```

The example defaults to `qal/data/RET_images/690_nm_LP.tiff`. Set `IMAGE_PATH`
at the top of the example to use a different TIFF.

The RET example offers `automatic_refinement` and `manual_coordinates` modes.
Both allow dragging the placed `+` before confirmation. Manual mode uses the
exact clicked or dragged signal coordinate and optionally accepts a
double-clicked background Control. Because one RET signal has no neighboring
signal from which to estimate its size, set `MANUAL_ROI_DIAMETER` explicitly.

Automatic mode requests exactly one approximate point:

```python
ret_roi = select_wells_gui(
    image,
    expected_count=1,
    well_ids=["RET"],
    detector=detector,
    search_radius=300,
    enable_live_grid_preview=True,
)
```

After Enter is pressed, the approximate click is matched to a detected well.
The resulting DataFrame contains the refined `x`, `y`, `ROI Diameter`, and
`ROI Radius`, making it directly compatible with `RetAnalyzer.analyze`.

![RET centroid and ROI found after confirmation](gui_images/ret_auto_detection.png)

## Concentration-target example

Run:

```bash
python -m qal.examples.concentration_rcs.concentration_gui_example
```

By default this loads `qal.data.cn_sample_3()`. Set `IMAGE_PATH` at the top of
the example to use a local TIFF, BMP, PNG, or JPEG.

Choose the product definition with `TARGET_TAG`:

```python
TARGET_TAG = "icg"   # ICG-equivalent, 1–1000 nM
# TARGET_TAG = "q800"  # Q800-01, 1–600 nM
# TARGET_TAG = "q700"  # Q700-01, 1–1000 nM
# TARGET_TAG = "custom"
```

The QUEL definitions are loaded from
`qal/data/product_presets/concentration.yaml`. Related depth, resolution,
uniformity/distortion, radiometric-emitter, and depth-resolution definitions
and reference-set bundle definitions are kept in the same `product_presets`
folder. For `"custom"`, edit the nine entries in `CUSTOM_WELL_IDS`; the final
entry must be `"Control"`. Catalog SKUs and unit serials can be resolved to
those presets with `python -m qal.examples.general.target_preset_lookup_example`.
SKU-driven GUI selection is
`python -m qal.examples.concentration_rcs.concentration_sku_gui_example`.

At the top of `qal/examples/concentration_rcs/concentration_gui_example.py`, choose one mode:

```python
MODE = "automatic_refinement"
# MODE = "manual_coordinates"
# MODE = "manual_live_preview"
```

`automatic_refinement` locally refines each click. `manual_coordinates` returns
only the exact points selected and does not infer missing grid positions.
`manual_live_preview` uses exact draggable anchors and completes the inferred
3×3 grid after confirmation.

Set `ANCHOR_COUNT` to a fixed value, or set it to `None` in
`manual_coordinates` mode to accept up to nine total regions. Anchors must be
selected in row-major order. The first three are the **top row from left to
right**:

1. `1000 nM`
2. `300 nM`
3. `100 nM`

The fourth anchor, when requested, is `60 nM` at the left of the second row.
Using four anchors lets the live-preview geometry fit use both grid axes.
Grid completion runs only for `automatic_refinement` and
`manual_live_preview`; `manual_coordinates` keeps four clicks as four rows.

With `ANCHOR_COUNT=None`, double-clicking a Control reserves the other eight
slots for signals. Without a double-clicked Control, the ninth ordinary click
is automatically labeled Control. Enter can confirm fewer than nine regions.
Control-dependent analysis is skipped only when no Control was selected; an
explicit Control enables baselining for a partial selection.

In manual live-preview mode, solid circles represent supplied anchors and
dashed circles represent inferred wells. Dragging any `+` immediately refits
the overlay.

![BMP manual coordinates with draggable grid preview](gui_images/bmp_manual_live_preview.png)

![Completed concentration target grid](gui_images/concentration_grid_completion.png)

The concentration labels are applied in row-major order:

```text
1000 nM | 300 nM | 100 nM
  60 nM |  30 nM |  10 nM
   3 nM |   1 nM | Control
```

The depth target uses the same geometry with:

```text
0.5 mm | 1.0 mm | 1.5 mm
2.0 mm | 3.0 mm | 4.0 mm
5.0 mm | 6.0 mm | Control
```

## Automatic coordinate refinement

With `coordinate_mode="detect"`, each click is refined independently by
`WellDetector.refine_well_at_coordinate`:

1. Crop a local window around the click.
2. Estimate background from the window border and seed intensity near the click.
3. Threshold relative to that local contrast and clean speckles with a small
   morphological opening.
4. Keep the connected component containing the seed (or the nearest nearby
   component).
5. Return that component's centroid and ROI diameter.

This local approach succeeds for low-limit-of-detection wells that full-image
`detect_wells` misses. Full-image detection still remains available for
unsupervised workflows; click refinement no longer depends on it.

`search_radius` is the local window half-width in image pixels. If omitted, it
defaults to 10% of the smaller image dimension. Use a larger value only when
clicks may be farther from the true center.

The automatic detector used by unsupervised `detect_wells` still performs:

1. Log-compress the image with `WellDetector.cvt_to_log_scale`.
2. Convert it to 8-bit with `WellDetector.cvt_to_uint8`.
3. Generate a series of threshold images with
   `WellDetector.get_thresh_rolling_window_binary`.
4. Find connected circular components and estimate their centroids and
   diameters.
5. Cluster repeated detections from different thresholds.
6. Sort the cluster centers by mean intensity.

## Authoritative manual coordinates

Use `coordinate_mode="manual"` when the wells are visible by eye but automatic
detection cannot reliably segment them:

```python
anchors = select_wells_gui(
    image,
    expected_count=4,
    well_ids=WELL_IDS[:4],
    coordinate_mode="manual",
    enable_live_grid_preview=True,
)
```

No call to `WellDetector.detect_wells` is made when the selection is
confirmed. The raw click or dragged coordinates become the anchor centroids.

`manual_roi_diameter` can set the ROI diameter in pixels. When it is `None`
and at least two signal anchors are available, the diameter is 60% of the
minimum pairwise distance among signal points. A double-clicked Control is
excluded from this calculation:

```python
anchors = select_wells_gui(
    image,
    expected_count=3,
    coordinate_mode="manual",
    manual_roi_diameter=100.0,
)
```

## Programmatic input without the GUI

The same refinement can be called directly:

```python
from qal import select_wells_from_coordinates

roi_df = select_wells_from_coordinates(
    image,
    [(1975, 1425)],
    well_ids=["RET"],
    search_radius=300,
)
```

Any number of coordinate pairs is accepted. In detection mode each point is
refined independently within its local search window.

To bypass detection:

```python
roi_df = select_wells_from_coordinates(
    image,
    [(332, 128), (507, 124), (681, 125), (330, 303)],
    well_ids=WELL_IDS[:4],
    coordinate_mode="manual",
    manual_roi_diameter=105,
)
```

Programmatic controls can be identified by coordinate index. Their coordinates
remain exact and they do not affect automatic diameter estimation:

```python
roi_df = select_wells_from_coordinates(
    image,
    [(332, 128), (507, 124), (681, 125), (50, 500)],
    well_ids=["1000 nM", "300 nM", "100 nM"],
    coordinate_mode="manual",
    control_indices=[3],
)
```

## Jupyter notebooks

`select_wells_gui` detects a Jupyter kernel and chooses an interactive
backend automatically.

- **JupyterLab / classic Notebook:** Matplotlib is switched to `ipympl` when
  needed. The picker is non-blocking and includes **Remove last** and
  **Confirm** buttons. After clicking Confirm, retrieve the DataFrame from
  `session.result`.
- **VS Code / Cursor:** Those editors do not register the
  `jupyter-matplotlib` widget (`MPLCanvasModel`). The picker opens a **native
  Matplotlib window** instead. Click the wells, press Enter, and
  `select_wells_gui` returns a DataFrame like the desktop examples.

```python
session = select_wells_gui(
    image,
    expected_count=1,
    well_ids=["RET"],
)

# JupyterLab: run after clicking Confirm.
# VS Code/Cursor: session is already a DataFrame.
if not hasattr(session, "result"):
    result = RetAnalyzer().analyze(image, session)
else:
    result = RetAnalyzer().analyze(image, session.result)
```

Desktop Matplotlib backends block while the GUI is open and return the
DataFrame directly after confirmation. If a notebook already has a desktop
GUI backend such as Qt, that same blocking window is used.

## Good to know

- Closing the desktop window before confirming raises an error.
- Enter (desktop) or Confirm (notebook) is rejected until
  `expected_count` signal points have been selected. An optional Control
  does not count toward that total.
- In detection mode, each click must land on or near a locally brighter well
  region. Manual mode has no such requirement.
- The current 3×3 completion minimum is three row-major anchors. The first
  three represent the top row in left-to-right order; a fourth represents the
  leftmost well of the second row.
- The completed 3×3 ROIs share the largest anchor radius.
- RET analysis accepts exactly one ROI.
- If automatic detection finds no suitable well near a click, increase
  `search_radius` only when the click was genuinely close to the intended
  well; otherwise inspect image contrast and detector output first.
