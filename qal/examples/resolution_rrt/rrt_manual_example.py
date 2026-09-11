import pandas as pd
from qal.data import resolution_target_cropped
from qal import RrtAnalyzer, RrtDataPlotter


def main():
    # Load the image of the resolution target
    im = resolution_target_cropped()

    # Define the group information with coordinates for each group
    group_info = {
        0: {'coordinates': ((64, 73), (64, 220)), 'elements': range(2, 7)},
        1: {'coordinates': ((199, 73), (199, 171)), 'elements': range(1, 7)},
        2: {'coordinates': ((131, 125), (131, 165)), 'elements': range(2, 7)},
    }

    # Create instances of the analyzer and visualizer
    analyzer = RrtAnalyzer()
    visualizer = RrtDataPlotter()

    # Process the groups and get the combined DataFrame
    print("Processing image data...")
    percentage_contrast_df = analyzer.load_and_process_groups(im, group_info)

    # Print the percentage contrast DataFrame
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(percentage_contrast_df)

    # Find the lp/mm, group, and element that comes right before 26%
    threshold = 26
    above_threshold_data = percentage_contrast_df[percentage_contrast_df['Percentage Contrast'] <= threshold]
    below_threshold_data = percentage_contrast_df[percentage_contrast_df['Percentage Contrast'] > threshold]

    if not below_threshold_data.empty:
        last_below_threshold = below_threshold_data.iloc[-1]
        lp_mm = last_below_threshold['Resolution (lp/mm)']
        group = last_below_threshold['Group']
        element = last_below_threshold['Element']
        print(f"\nSmallest feature resolved with a percentage contrast above 26%:")
        print(f"Resolution: {lp_mm:.2f} lp/mm")
        print(f"Group: {int(group)}")
        print(f"Element: {element}")
    else:
        print("All data points are below 26% contrast.")

    # Plot the percentage contrast values
    visualizer.plot_percentage_contrast(percentage_contrast_df)
    visualizer.plot_line_profiles(im, group_info, percentage_contrast_df)

if __name__ == '__main__':
    main()