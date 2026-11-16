"""Analyze a RET light/dark pair automatically, without manual selection.

Set DARK_FRAME_PATH = None for light-frame-only analysis. Edit the settings
below, then run:

    python -m qal.examples.radiometric_emitter_ret.ret_light_frame_automatic_analysis_example
"""

from qal.data import ret_sample_1_df, ret_sample_1_lf
from qal.examples.radiometric_emitter_ret import ret_gui_example as workflow

IMAGE_PATH = ret_sample_1_lf()
DARK_FRAME_PATH = ret_sample_1_df()
# IMAGE_PATH = "path/to/light_frame.tiff"
# DARK_FRAME_PATH = "path/to/dark_frame.tiff"
# DARK_FRAME_PATH = None

SAVE_RESULTS = True
SAVE_SOURCE_IMAGES = False
REGION_OF_WELL_TO_ANALYZE = 0.5
CNR_THRESHOLD = 3.0


def main():
    return workflow.run_ret_example(
        IMAGE_PATH, DARK_FRAME_PATH, mode="automatic_detection",
        save_results=SAVE_RESULTS, save_sources=SAVE_SOURCE_IMAGES,
        output_fallback=__file__, region=REGION_OF_WELL_TO_ANALYZE,
        cnr_threshold=CNR_THRESHOLD,
    )


if __name__ == "__main__":
    main()
