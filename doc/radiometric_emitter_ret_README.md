# RET single-well analysis

The RET workflow follows the same separation used by other QAL products:

1. `WellDetector` extracts and selects the single circular RET ROI.
2. `RetAnalyzer` measures the supplied ROI.
3. `WellDetector.plot_detected_wells` can visualize the extracted ROI.

A measured dark frame is the default production baseline. The dark frame is
not subtracted pixel-by-pixel: the extracted signal ROI coordinates and size
are transferred to the dark frame, and that matched ROI supplies the baseline
statistics.

## Analysis with a dark frame

The signal image and dark frame must have the same dimensions.

```python
from qal import RetAnalyzer, WellDetector
from qal.util import load_grayscale_image

image = load_grayscale_image("ret_signal.tiff")
detector = WellDetector(parallel_processing=False)
detections = detector.detect_wells(image, well_ids=["RET"])
ret_roi = detector.select_single_well(image, detections)

result = RetAnalyzer().analyze(
    image,
    ret_roi,
    dark_frame="dark_frame.tiff",
)
print(result.stats_df)
```

The baselined mean and contrast-to-noise ratio (CNR) are:

```text
baselined mean = signal ROI mean - dark-frame ROI mean
CNR = baselined mean / dark-frame ROI standard deviation
```

If the dark-frame ROI has zero standard deviation, CNR is reported as `NaN`
and the threshold check fails.

## Raw analysis without a dark frame

The same extracted ROI can be measured without a dark frame:

```python
result = RetAnalyzer().analyze(image, ret_roi)
print(result.stats_df)
```

This result contains the ROI location, signal mean, and signal standard
deviation. It does not contain a fabricated baseline, normalized output, CNR,
or CNR threshold result.

## Configuration

```python
from qal import RetAnalysisConfig, RetAnalyzer

analyzer = RetAnalyzer(
    analysis_config=RetAnalysisConfig(
        region_of_well_to_analyze=0.5,
        cnr_threshold=3.0,
    ),
)
```

`region_of_well_to_analyze` is the fraction of the detected well diameter used
for measurement. If multiple circles are detected, select one before analysis:

```python
ret_roi = detector.select_single_well(
    image,
    detections,
    selection="brightest",
)
```

## Pseudo-dark testing

The detector contains an explicit test helper for extracting a pseudo-dark
ROI. The analyzer only measures the ROIs it receives:

```python
pseudo_dark_roi = detector.get_pseudo_dark_roi_for_testing(image, ret_roi)
result = analyzer.analyze_with_pseudo_dark_for_testing(
    image,
    ret_roi,
    pseudo_dark_roi,
)
```

This analysis emits a runtime warning and labels the baseline as
`pseudo_dark_test_only`. Normal `analyze(...)` never falls back to this
behavior. Pseudo-dark results must not be used for production RET analysis.

Runnable scripts are available in `qal/examples/radiometric_emitter_ret/`:

- `ret_gui_example.py`: automatic refinement (default), manual coordinates, or
  automatic detection. Manual mode also supports a separately reported background
  estimate. Uncomment `MANUAL_ROI_DIAMETER` before selecting manual mode.
- `ret_light_frame_automatic_analysis_example.py`: automatic detection and
  analysis with no well-selection window.

Both scripts default to sample 1's matching light/dark pair and save `stats.csv`
and `roi.png`. Set `DARK_FRAME_PATH = None` for raw light-frame-only analysis.
`ret_dark_frame_example.py` has been consolidated into `ret_gui_example.py`.
Coordinates are included in `stats.csv`, so there is no duplicate `wells.csv`.
For pair 2, use `qal.data.ret_sample_2_lf()` and `qal.data.ret_sample_2_df()` together.
All four loaders resolve local checkout data or download registered,
SHA-256-validated images from the selected repository and branch.

Results go beside a user-supplied light frame, or beside the script for built-in
samples. `SAVE_SOURCE_IMAGES = False` keeps source copying off by default.
Enable it to copy original built-in files without re-encoding them or relocating
cached data. Two source frames are grouped in `source_images/`; user-supplied
files are not copied. `SAVE_RESULTS = False` disables all exports.

For the popup picker with inline plots and optional CSV/plot saving, open
`qal/examples/jupyter-notebook/radiometric_emitter_ret_gui_example.ipynb`.
It defaults to sample 1's matching light/dark pair, including dark-frame baseline
and CNR statistics. Set `DARK_FRAME_PATH = None` for raw analysis only.

The well-selection GUI guide is `doc/well_detector_gui_README.md`.

A two-click line tool for estimating manual ROI diameter remains a proposed follow-up.
