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

- `ret_dark_frame_example.py`
- `ret_light_frame_automatic_analysis_example.py`
- `ret_gui_example.py`

The well-selection GUI guide is `qal/examples/general/well_detector_gui_README.md`.
