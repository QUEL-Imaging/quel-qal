from qal.data import cn_sample_3, depth_sample_1
from qal import WellDetector, WellAnalyzer, WellPlotter
import pandas as pd

def main():
    ### Concentration Target Example ###

    # Initialize WellDetector
    detector = WellDetector()

    # Step 1: Detect wells in Concentration Target
    cn_im = cn_sample_3()
    print("Processing Concentration Target...")

    cn_df = detector.detect_wells(
        cn_im, show_detected_wells=True, debug=False, set_consistent_roi_region=True
    )
    cn_df = detector.estimate_remaining_wells_3x3(
        cn_im, cn_df, well_ids=[
            '1000 nM', '300 nM', '100 nM', '60 nM', '30 nM', '10 nM', '3 nM', '1 nM', 'Control'
        ]
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Concentration Target ROI:")
        print(cn_df)

    # Step 2: Analyze the ROIs found in step 1
    print("\nAnalyzing Concentration Target...")
    analyzer = WellAnalyzer(cn_im, cn_df)
    cn_df = analyzer.get_stats(region_of_well_to_analyze=0.5)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(cn_df)

    # Step 3: Plot the detected wells and well intensity graphs
    print("\nVisualizing Concentration Target ROI...")
    plotter = WellPlotter(cn_df, image=cn_im)
    plotter.visualize_roi()

    print("\nPlotting Concentration Target Results...")
    plotter.plot(
        graph_type='concentration',
        col_to_plot='mean intensity normalized',
        plot_error_bars=False,
        save_plot=None
    )

    ### Depth Target Example ###

    # Step 1: Detect wells in Depth Target
    depth_im = depth_sample_1()
    print("\nProcessing Depth Target...")

    depth_df = detector.detect_wells(
        depth_im, show_detected_wells=True, debug=False, set_consistent_roi_region=True
    )
    depth_df = detector.estimate_remaining_wells_3x3(
        depth_im, depth_df, well_ids=[
            '0.5 mm', '1.0 mm', '1.5 mm', '2.0 mm', '3.0 mm', '4.0 mm', '5.0 mm', '6.0 mm', 'Control'
        ]
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("\nDepth Target ROI:")
        print(depth_df)

    # Step 2: Analyze the ROIs found in step 1
    print("\nAnalyzing Depth Target...")
    analyzer = WellAnalyzer(depth_im, depth_df)
    depth_df = analyzer.get_stats(region_of_well_to_analyze=0.5)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(depth_df)

    # Step 3: Plot the detected wells and well intensity graphs
    print("\nVisualizing Depth Target ROI...")
    plotter = WellPlotter(depth_df, image=depth_im)
    plotter.visualize_roi()

    print("\nPlotting Depth Target Results...")
    plotter.plot(
        graph_type='depth',
        col_to_plot='mean intensity normalized',
        plot_error_bars=False,
        trendline_lib='scipy',
        save_plot=None
    )

if __name__ == "__main__":
    main()
