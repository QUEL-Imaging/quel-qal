import numpy as np
import pandas as pd
import pytest

from qal import RetAnalysisConfig, RetAnalyzer, WellDetector
from qal.rta.image_analyzer.ret_analyzer import summarize_ret_results
from qal.util import circular_mask, mean_in_circular_roi, std_in_circular_roi


def detection(x=10.0, y=10.0, diameter=8.0):
    return pd.DataFrame(
        [
            {
                "x": x,
                "y": y,
                "ROI Diameter": diameter,
                "ROI Radius": diameter / 2,
            }
        ]
    )


def test_dark_frame_uses_matched_signal_roi():
    image = np.full((21, 21), 5.0)
    dark = np.zeros_like(image)
    mask = circular_mask(dark.shape, (10, 10), 2)
    dark[mask] = np.arange(mask.sum()) % 3
    image[mask] = 20.0

    result = RetAnalyzer().analyze(
        image,
        detection(),
        dark_frame=dark,
    )

    row = result.stats_df.iloc[0]
    expected_dark_mean = mean_in_circular_roi(dark, (10, 10), 2)
    expected_dark_std = std_in_circular_roi(dark, (10, 10), 2)
    assert result.ok
    assert result.is_baselined
    assert row["baseline source"] == "dark_frame"
    assert row["dark frame x"] == 10
    assert row["dark frame y"] == 10
    assert row["dark frame mean intensity"] == pytest.approx(expected_dark_mean)
    assert row["dark frame standard deviation"] == pytest.approx(expected_dark_std)
    assert row["mean intensity baselined"] == pytest.approx(
        20 - expected_dark_mean
    )
    assert row["CNR"] == pytest.approx(
        (20 - expected_dark_mean) / expected_dark_std
    )


def test_analysis_without_dark_frame_is_raw_only():
    result = RetAnalyzer().analyze(
        np.full((21, 21), 12.0),
        detection(),
    )

    assert result.ok
    assert not result.is_baselined
    assert result.stats_df.loc[0, "mean intensity"] == 12
    assert "dark frame mean intensity" not in result.stats_df
    assert "mean intensity baselined" not in result.stats_df
    assert "CNR" not in result.stats_df
    assert "CNR pass" not in result.stats_df


def test_dark_frame_shape_must_match_signal():
    with pytest.raises(ValueError, match="must match"):
        RetAnalyzer().analyze(
            np.zeros((20, 20)),
            detection(),
            dark_frame=np.zeros((10, 10)),
        )


def test_zero_dark_noise_produces_nan_cnr_and_failed_threshold():
    result = RetAnalyzer().analyze(
        np.full((21, 21), 12.0),
        detection(),
        dark_frame=np.full((21, 21), 2.0),
    )

    assert np.isnan(result.stats_df.loc[0, "CNR"])
    assert not result.stats_df.loc[0, "CNR pass"]


def test_pseudo_dark_requires_explicit_testing_method():
    yy, xx = np.indices((80, 80))
    image = (xx + yy).astype(float)
    image[circular_mask(image.shape, (40, 40), 4)] += 100
    detector = WellDetector(parallel_processing=False)
    ret_roi = detection(x=40, y=40, diameter=8)
    pseudo_dark_roi = detector.get_pseudo_dark_roi_for_testing(
        image,
        ret_roi,
        region_of_well_to_analyze=1.0,
    )

    with pytest.warns(RuntimeWarning, match="testing only"):
        result = RetAnalyzer(
            analysis_config=RetAnalysisConfig(region_of_well_to_analyze=1.0)
        ).analyze_with_pseudo_dark_for_testing(
            image,
            ret_roi,
            pseudo_dark_roi,
        )

    assert result.stats_df.loc[0, "baseline source"] == "pseudo_dark_test_only"
    assert result.dark_frame is None
    assert result.is_baselined


def test_summarize_results_combines_analyzer_outputs():
    analyzer = RetAnalyzer()
    images = [np.full((21, 21), value) for value in (10.0, 20.0)]
    dark = np.arange(21 * 21, dtype=float).reshape(21, 21)
    results = [
        analyzer.analyze(image, detection(), dark_frame=dark)
        for image in images
    ]

    assert len(results) == 2
    assert all(result.ok and result.is_baselined for result in results)
    assert len(summarize_ret_results(results)) == 2


def test_analyzer_requires_one_previously_extracted_roi():
    analyzer = RetAnalyzer()
    with pytest.raises(ValueError, match="must contain one extracted well"):
        analyzer.analyze(np.zeros((20, 20)), pd.DataFrame())
    with pytest.raises(ValueError, match="exactly one well"):
        analyzer.analyze(
            np.zeros((20, 20)),
            pd.concat([detection(), detection()], ignore_index=True),
        )


def test_well_detector_selects_brightest_single_well():
    image = np.zeros((30, 30), dtype=float)
    image[circular_mask(image.shape, (8, 8), 3)] = 5
    image[circular_mask(image.shape, (22, 22), 3)] = 20
    detections = pd.concat(
        [
            detection(x=8, y=8, diameter=6),
            detection(x=22, y=22, diameter=6),
        ],
        ignore_index=True,
    )

    selected = WellDetector(
        parallel_processing=False
    ).select_single_well(image, detections)

    assert len(selected) == 1
    assert selected.iloc[0]["x"] == 22
    assert selected.iloc[0]["y"] == 22
