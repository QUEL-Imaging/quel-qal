from qal.data import res_sample_1
from qal import RrtROI, RrtAnalyzer, RrtDataPlotter
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Create an instance of RrtROI
    rrt_roi = RrtROI()

    # Load the image of the resolution target
    image = res_sample_1()

    # Check the environment (Jupyter Notebook or standard Python)
    environment = rrt_roi.check_environment()

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Original Image')
    fig.suptitle('Original Image', fontsize=16)
    plt.imshow(image, cmap='gray')
    plt.show()

    # Crop the resolution target
    cropped_image = rrt_roi.get_resolution_target_cropped(image, show_kp=False)

    # Display cropped image
    if cropped_image is not None:
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Cropped Image')
        fig.suptitle('Cropped Image', fontsize=16)
        plt.imshow(cropped_image, cmap='gray')
        plt.show()

    # Semi-automatic selection of resolution target corners
    if "Jupyter" in environment:
        rrt_roi.select_points_jupyter(cropped_image)
    else:
        rrt_roi.select_points_standard(cropped_image)

    # Print the detected points
    if rrt_roi.roi_corners is not None:
        print("\nClosest corners detected:")
        for corner in rrt_roi.roi_corners:
            print(f"(x={corner[1]:.2f}, y={corner[0]:.2f})")

    # Create instances of RrtAnalyzer and RrtDataPlotter
    analyzer = RrtAnalyzer()
    visualizer = RrtDataPlotter()

    # Define the group information based on the closest corners detected
    group_coordinates = rrt_roi.group_coordinates

    # Process the groups and get the combined DataFrame
    percentage_contrast_df = analyzer.load_and_process_groups(cropped_image, group_coordinates)

    # Print the percentage contrast DataFrame
    print("\nPercentage Contrast DataFrame:")
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
        print("\nAll data points are below 26% contrast.")

    # Plot the percentage contrast values
    visualizer.plot_line_profiles(cropped_image, group_coordinates, percentage_contrast_df)
    visualizer.plot_percentage_contrast(percentage_contrast_df)

if __name__ == '__main__':
    main()