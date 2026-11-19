# Depth-resolution examples

Run `python -m qal.examples.depth_resolution_dr.depth_resolution_example`.
One sample runs at a time. Choose one `IMAGE_PATH` assignment; comment out the
others. Select top, bottom, left, and right edges, then press Enter or close the
window to confirm. The cropped view accepts the same confirmation actions.

| Input | Configuration | Expected comparison |
| --- | --- | --- |
| `dr_sample1()` | `CUSTOM_DIMENSIONS = False` | Standard dimensions, profiles at ten depths from 1 to 6 mm. |
| `dr_sample2()` | `CUSTOM_DIMENSIONS = False` | Channel intensity drops below 2%; the FWHM plot includes an additional indicator. |
| `dr_sample3()` | `CUSTOM_DIMENSIONS = True` | Custom 35 × 50 mm phantom, depth range 1.3–7.3 mm, descent from 6–44 mm, channel 0.5 mm from the top. |

Sample 3 also exposes `ROTATE_IMAGE_90` for images needing a quarter turn.
The Matplotlib backend uses the same platform-aware native-window helper as
the other GUI examples; macOS no longer unconditionally selects TkAgg.

Plots save by default beside the script for built-in arrays, or beside your
image file for user paths. `SAVE_SOURCE_IMAGES` optionally copies the original
built-in input. It is off by default and does not copy user-supplied files.

This comparison is suitable for a wiki entry. Measured results and comparison
figures should be added after selecting equivalent phantom borders in all three
samples; this change does not fabricate measurements from unconfirmed crops.
