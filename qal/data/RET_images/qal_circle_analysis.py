#!/usr/bin/env python3
"""Single-circle filter-check analysis using quel-qal WellDetector / WellAnalyzer."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from skimage import io
from qal import WellAnalyzer, WellDetector, WellPlotter

FOLDER = Path(__file__).resolve().parent
OUT = FOLDER / "qal_circle_analysis_output"
REGION_FRACTION = 0.5  # quel-qal default: analyze inner 50% of well diameter
CNR_THRESHOLD = 3.0  # AAPM TG-311 recommendation used by quel-qal


def load_meta(tiff_path: Path) -> dict:
    yaml_path = tiff_path.with_suffix(".yaml")
    if yaml_path.exists():
        with yaml_path.open() as f:
            return yaml.safe_load(f)
    return {}


def pick_brightest_well(df: pd.DataFrame, im: np.ndarray) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    detector = WellDetector(parallel_processing=False)
    scored = detector.get_well_intensities(im, df.copy(), sort_by_intensity=True)
    return scored.iloc[[0]].copy()


def make_control_row(im: np.ndarray, well: pd.Series) -> pd.Series:
    """Build a Control well using a same-size ROI in a dark corner, matching quel-qal's CNR definition."""
    radius = float(well["ROI Radius"])
    cy, cx = float(well["y"]), float(well["x"])
    h, w = im.shape[:2]
    margin = max(int(2.5 * radius), 40)
    candidates = [
        (margin, margin),
        (margin, w - margin),
        (h - margin, w - margin),
        (h - margin, margin),
        (h // 2, margin),
        (h // 2, w - margin),
        (margin, w // 2),
        (h - margin, w // 2),
    ]
    best = None
    best_mean = np.inf
    for y, x in candidates:
        if y < radius or x < radius or y >= h - radius or x >= w - radius:
            continue
        if np.hypot(x - cx, y - cy) < 4 * radius:
            continue
        Y, X = np.ogrid[:h, :w]
        mask = np.sqrt((X - x) ** 2 + (Y - y) ** 2) <= radius
        mean = float(np.mean(im[mask]))
        if mean < best_mean:
            best_mean = mean
            best = (x, y)
    if best is None:
        best = (margin, margin)
        best_mean = float(np.mean(im[: 2 * margin, : 2 * margin]))
    row = well.copy()
    row["x"] = float(best[0])
    row["y"] = float(best[1])
    row["well"] = "Control"
    row["value"] = 0.0
    return row


def radial_profile(im: np.ndarray, cx: float, cy: float, max_r: float, n_bins: int = 80) -> dict:
    h, w = im.shape[:2]
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = r <= max_r
    radii = r[mask]
    vals = im[mask].astype(float)
    bins = np.linspace(0, max_r, n_bins + 1)
    idx = np.clip(np.digitize(radii, bins) - 1, 0, n_bins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.array([vals[idx == i].mean() if np.any(idx == i) else np.nan for i in range(n_bins)])
    stds = np.array([vals[idx == i].std() if np.any(idx == i) else np.nan for i in range(n_bins)])
    counts = np.array([int(np.sum(idx == i)) for i in range(n_bins)])
    return {
        "radius_px": centers.tolist(),
        "mean": np.nan_to_num(means, nan=0.0).tolist(),
        "std": np.nan_to_num(stds, nan=0.0).tolist(),
        "counts": counts.tolist(),
    }


def line_profiles(im: np.ndarray, cx: float, cy: float, half_width: int) -> dict:
    h, w = im.shape[:2]
    cxi, cyi = int(round(cx)), int(round(cy))
    x0 = max(0, cxi - half_width)
    x1 = min(w, cxi + half_width + 1)
    y0 = max(0, cyi - half_width)
    y1 = min(h, cyi + half_width + 1)
    horiz = im[cyi, x0:x1].astype(float)
    vert = im[y0:y1, cxi].astype(float)
    return {
        "horizontal_x_px": list(range(x0, x1)),
        "horizontal": horiz.tolist(),
        "vertical_y_px": list(range(y0, y1)),
        "vertical": vert.tolist(),
    }


def rise_width(radius: np.ndarray, mean: np.ndarray, low_frac: float = 0.1, high_frac: float = 0.9) -> dict:
    valid = np.isfinite(mean)
    r, m = radius[valid], mean[valid]
    if m.size < 5:
        return {"r10_px": None, "r90_px": None, "rise_10_90_px": None}
    bg = float(np.nanmedian(m[-max(5, m.size // 8) :]))
    peak = float(np.nanmax(m[: max(5, m.size // 3)]))
    span = peak - bg
    if span <= 0:
        return {"r10_px": None, "r90_px": None, "rise_10_90_px": None, "peak": peak, "bg": bg}
    t10, t90 = bg + low_frac * span, bg + high_frac * span
    # walk from outside in
    r10 = r90 = None
    for ri, mi in zip(r[::-1], m[::-1]):
        if r10 is None and mi >= t10:
            r10 = float(ri)
        if r90 is None and mi >= t90:
            r90 = float(ri)
        if r10 is not None and r90 is not None:
            break
    rise = (r10 - r90) if (r10 is not None and r90 is not None) else None
    return {"r10_px": r10, "r90_px": r90, "rise_10_90_px": rise, "peak": peak, "bg": bg}


def saturation_frac(im: np.ndarray, cx: float, cy: float, radius: float, bit_depth: int) -> float:
    h, w = im.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) <= radius
    sat = (2 ** bit_depth) - 1
    # ASI cameras often map 12-bit into 16-bit containers; also check 16-bit ceiling
    vals = im[mask]
    if vals.size == 0:
        return 0.0
    return float(np.mean((vals >= sat * 0.98) | (vals >= 0.98 * 65535)))


def preview_image(im: np.ndarray, p_low: float = 1, p_high: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(im, (p_low, p_high))
    if hi <= lo:
        hi = lo + 1
    vis = np.clip((im.astype(float) - lo) / (hi - lo), 0, 1)
    return vis


def plot_overlay(im: np.ndarray, stats_df: pd.DataFrame, title: str, path: Path) -> None:
    vis = preview_image(im)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.imshow(vis, cmap="gray")
    colors = {"Circle": "#3b82f6", "Control": "#ef4444"}
    for _, row in stats_df.iterrows():
        color = colors.get(str(row["well"]), "#22c55e")
        ax.add_patch(
            plt.Circle(
                (row["x"], row["y"]),
                row["Analyzed ROI Diameter"] / 2,
                edgecolor=color,
                facecolor="none",
                lw=2,
                label=f"{row['well']} analysis ROI",
            )
        )
        ax.add_patch(
            plt.Circle(
                (row["x"], row["y"]),
                row["ROI Radius"],
                edgecolor=color,
                facecolor="none",
                lw=1,
                linestyle="--",
                alpha=0.8,
            )
        )
        ax.plot(row["x"], row["y"], "+", color=color, markersize=10)
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(filter_name: str, radial: dict, lines: dict, roi_radius: float, analyzed_r: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    r = np.array(radial["radius_px"])
    m = np.array(radial["mean"])
    s = np.array(radial["std"])
    axes[0].plot(r, m, color="#1B3D87", lw=2, label="Azimuthal mean")
    axes[0].fill_between(r, m - s, m + s, color="#1B3D87", alpha=0.2, label="±1 σ")
    axes[0].axvline(analyzed_r, color="#16a34a", ls="--", lw=1.2, label=f"Analysis ROI r={analyzed_r:.1f} px")
    axes[0].axvline(roi_radius, color="#b45309", ls=":", lw=1.2, label=f"Detected r={roi_radius:.1f} px")
    axes[0].set_xlabel("Radius from centroid (px)")
    axes[0].set_ylabel("Intensity (ADU)")
    axes[0].set_title(f"{filter_name} radial profile")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    hx = np.array(lines["horizontal_x_px"])
    hy = np.array(lines["horizontal"])
    vx = np.array(lines["vertical_y_px"])
    vy = np.array(lines["vertical"])
    axes[1].plot(hx, hy, color="#1B3D87", lw=1.5, label="Horizontal")
    axes[1].plot(vx, vy, color="#dc2626", lw=1.5, alpha=0.85, label="Vertical")
    axes[1].set_xlabel("Pixel coordinate")
    axes[1].set_ylabel("Intensity (ADU)")
    axes[1].set_title(f"{filter_name} center line profiles")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def inspect_image(path: Path, im: np.ndarray, meta: dict) -> dict:
    return {
        "file": path.name,
        "shape": list(im.shape),
        "dtype": str(im.dtype),
        "min": int(im.min()),
        "max": int(im.max()),
        "mean": float(im.mean()),
        "median": float(np.median(im)),
        "p99": float(np.percentile(im, 99)),
        "p999": float(np.percentile(im, 99.9)),
        "filter": (meta.get("filter_wheel") or {}).get("actual_filter_name"),
        "ret_level": (meta.get("ret") or {}).get("commanded_level"),
        "exposure_ms": (meta.get("camera") or {}).get("exposure_ms"),
        "gain": (meta.get("camera") or {}).get("gain"),
        "bit_depth": (meta.get("camera") or {}).get("BitDepth"),
        "temperature_C": (meta.get("camera") or {}).get("temperature_C"),
    }


def analyze_one(tiff_path: Path) -> dict:
    print(f"\n=== {tiff_path.name} ===")
    im = io.imread(str(tiff_path))
    if im.ndim == 3:
        im = im[:, :, 0] if im.shape[2] in (3, 4) else im[0]
    meta = load_meta(tiff_path)
    info = inspect_image(tiff_path, im, meta)
    print(
        f"  shape={info['shape']} dtype={info['dtype']} "
        f"min/max/mean={info['min']}/{info['max']}/{info['mean']:.1f} "
        f"filter={info['filter']} RET={info['ret_level']}"
    )

    detector = WellDetector(parallel_processing=True)
    try:
        df = detector.detect_wells(
            im,
            well_ids=["Circle"],
            show_detected_wells=False,
            debug=False,
            set_consistent_roi_region=True,
            downscale=0.5,
        )
    except Exception as exc:
        print(f"  detect_wells failed: {exc}")
        traceback.print_exc()
        df = None

    n_detected = 0 if df is None or df.empty else len(df)
    print(f"  wells detected: {n_detected}")
    if df is not None and not df.empty:
        print(df.to_string())

    if df is None or df.empty:
        return {
            "ok": False,
            "error": "WellDetector did not find a circular well",
            "image": info,
            "n_detected": 0,
        }

    well_df = pick_brightest_well(df.reset_index(drop=True), im)
    well_df["well"] = "Circle"
    well_df["value"] = 1.0
    control = make_control_row(im, well_df.iloc[0])
    analysis_df = pd.concat([well_df, pd.DataFrame([control])], ignore_index=True)
    for col in ("mean_intensity", "cluster"):
        if col in analysis_df.columns:
            analysis_df = analysis_df.drop(columns=[col])

    analyzer = WellAnalyzer(im, analysis_df)
    stats = analyzer.get_stats(region_of_well_to_analyze=REGION_FRACTION)
    print("  WellAnalyzer stats:")
    print(stats.to_string())

    circle = stats[stats["well"] == "Circle"].iloc[0]
    control_row = stats[stats["well"] == "Control"].iloc[0]
    cx, cy = float(circle["x"]), float(circle["y"])
    roi_r = float(circle["ROI Radius"])
    analyzed_r = float(circle["Analyzed ROI Diameter"]) / 2.0
    radial = radial_profile(im, cx, cy, max_r=max(2.5 * roi_r, analyzed_r * 3), n_bins=90)
    lines = line_profiles(im, cx, cy, half_width=int(max(2.2 * roi_r, 80)))
    edge = rise_width(np.array(radial["radius_px"]), np.array(radial["mean"]))
    bit_depth = int(info["bit_depth"] or 12)
    sat = saturation_frac(im, cx, cy, analyzed_r, bit_depth)
    cv = float(circle["standard deviation"] / circle["mean intensity"]) if circle["mean intensity"] else None
    snr = float(circle["mean intensity"] / circle["standard deviation"]) if circle["standard deviation"] else None

    overlay_path = OUT / f"{tiff_path.stem}_roi.png"
    profile_path = OUT / f"{tiff_path.stem}_profiles.png"
    plot_overlay(im, stats, f"{info['filter'] or tiff_path.stem} — quel-qal ROI", overlay_path)
    plot_profiles(info["filter"] or tiff_path.stem, radial, lines, roi_r, analyzed_r, profile_path)

    roi_plotter = WellPlotter(stats, image=im)
    roi_vis_path = OUT / f"{tiff_path.stem}_wellplotter_roi.png"
    try:
        roi_plotter.visualize_roi()
        fig = plt.gcf()
        fig.savefig(roi_vis_path, dpi=140, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:
        print(f"  WellPlotter.visualize_roi failed: {exc}")
        roi_vis_path = None

    result = {
        "ok": True,
        "image": info,
        "n_detected": n_detected,
        "detection": {
            "x_px": cx,
            "y_px": cy,
            "roi_diameter_px": float(circle["ROI Diameter"]),
            "roi_radius_px": roi_r,
            "analyzed_roi_diameter_px": float(circle["Analyzed ROI Diameter"]),
            "analyzed_roi_radius_px": analyzed_r,
        },
        "circle": {
            "mean_intensity_adu": float(circle["mean intensity"]),
            "std_adu": float(circle["standard deviation"]),
            "cv": cv,
            "snr_mean_over_std": snr,
            "mean_intensity_baselined_adu": float(circle["mean intensity baselined"]),
            "mean_intensity_normalized": float(circle["mean intensity normalized"]),
            "cnr": float(circle["CNR"]),
            "cnr_pass_tg311": bool(circle["CNR"] >= CNR_THRESHOLD),
        },
        "control": {
            "x_px": float(control_row["x"]),
            "y_px": float(control_row["y"]),
            "mean_intensity_adu": float(control_row["mean intensity"]),
            "std_adu": float(control_row["standard deviation"]),
        },
        "profile": {
            **edge,
            "saturation_frac_in_analysis_roi": sat,
        },
        "radial": {
            "radius_px": [round(v, 2) for v in radial["radius_px"]],
            "mean": [round(v, 2) for v in radial["mean"]],
        },
        "line_profiles": {
            "horizontal_peak_adu": float(np.max(lines["horizontal"])),
            "vertical_peak_adu": float(np.max(lines["vertical"])),
            "horizontal_center_adu": float(im[int(round(cy)), int(round(cx))]),
        },
        "files": {
            "overlay": overlay_path.name,
            "profiles": profile_path.name,
            "wellplotter_roi": None if roi_vis_path is None else roi_vis_path.name,
        },
    }
    return result


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tiffs = sorted(FOLDER.glob("*.tiff")) + sorted(FOLDER.glob("*.tif"))
    if not tiffs:
        raise SystemExit(f"No TIFF files in {FOLDER}")
    print(f"quel-qal single-circle analysis")
    print(f"folder: {FOLDER}")
    print(f"images: {[p.name for p in tiffs]}")

    results = []
    for path in tiffs:
        results.append(analyze_one(path))

    ok = [r for r in results if r.get("ok")]
    summary = {
        "library": "quel-qal 0.2.5",
        "method": {
            "detection": "WellDetector.detect_wells (log-compress, 8-bit stretch, rolling-window threshold, regionprops, agglomerative clustering)",
            "quantification": "WellAnalyzer.get_stats with region_of_well_to_analyze=0.5",
            "cnr": "(mean_circle - mean_control) / std_control; Control is a same-size background ROI (no Control well on target)",
            "cnr_threshold": CNR_THRESHOLD,
            "cnr_threshold_source": "AAPM Task Group 311, as used by quel-qal",
        },
        "n_images": len(results),
        "n_ok": len(ok),
        "results": results,
    }
    if ok:
        means = [r["circle"]["mean_intensity_adu"] for r in ok]
        cnrs = [r["circle"]["cnr"] for r in ok]
        brightest = max(ok, key=lambda r: r["circle"]["mean_intensity_adu"])
        dimmest = min(ok, key=lambda r: r["circle"]["mean_intensity_adu"])
        summary["comparison"] = {
            "mean_intensity_range_adu": [float(min(means)), float(max(means))],
            "mean_intensity_ratio_max_over_min": float(max(means) / min(means)) if min(means) else None,
            "cnr_range": [float(min(cnrs)), float(max(cnrs))],
            "brightest_filter": brightest["image"]["filter"],
            "dimmest_filter": dimmest["image"]["filter"],
            "all_cnr_pass_tg311": all(r["circle"]["cnr_pass_tg311"] for r in ok),
        }

    out_json = OUT / "analysis_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    if ok:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        labels = [r["image"]["filter"] or r["image"]["file"] for r in ok]
        means = [r["circle"]["mean_intensity_adu"] for r in ok]
        cnrs = [r["circle"]["cnr"] for r in ok]
        axes[0].bar(labels, means, color="#1B3D87")
        axes[0].set_ylabel("Mean intensity (ADU)")
        axes[0].set_title("Circle mean intensity by filter")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(labels, cnrs, color="#0f766e")
        axes[1].axhline(CNR_THRESHOLD, color="#b45309", ls="--", label="CNR = 3 (TG-311)")
        axes[1].set_ylabel("CNR")
        axes[1].set_title("Contrast-to-noise ratio by filter")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "filter_comparison.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        for r in ok:
            ax.plot(
                r["radial"]["radius_px"],
                r["radial"]["mean"],
                lw=2,
                label=r["image"]["filter"] or r["image"]["file"],
            )
        ax.set_xlabel("Radius from centroid (px)")
        ax.set_ylabel("Azimuthal mean intensity (ADU)")
        ax.set_title("Radial intensity profiles by emission filter")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "radial_profiles_overlay.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
