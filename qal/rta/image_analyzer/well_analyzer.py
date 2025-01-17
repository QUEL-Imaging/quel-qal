import numpy as np
import pandas as pd

class WellAnalyzer:
    def __init__(self, im, df):
        self.im = im
        self.df = df

    def get_normalized_column(self, col_name, base_col_name=None):
        if base_col_name is None:
            base_col_name = col_name

        min_val = self.df[base_col_name].min()
        max_val = self.df[base_col_name].max()

        self.df[f"{col_name} normalized"] = (self.df[col_name] - min_val) / (max_val - min_val)

    def circular_mask(self, center, radius):
        shape = self.im.shape
        Y, X = np.ogrid[:shape[0], :shape[1]]
        distance_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = distance_from_center <= radius
        im_copy = self.im.copy()
        im_copy = im_copy.astype(float)
        im_copy[~mask] = np.nan
        return im_copy

    def get_mean_intensity(self, im, idx):
        self.df.loc[idx, 'mean intensity'] = np.nanmean(im)

    def get_mean_intensity_baselined(self):
        value_to_subtract = self.df[self.df['well'] == 'Control']['mean intensity'].values[0]
        self.df['mean intensity baselined'] = self.df['mean intensity'] - value_to_subtract

    def get_standard_deviation(self, im, idx):
        self.df.loc[idx, 'standard deviation'] = np.nanstd(im[im != np.inf])

    def get_normalized_std(self):
        control_mean = self.df[self.df['well'] == 'Control']['mean intensity'].values[0]
        control_std = self.df[self.df['well'] == 'Control']['standard deviation'].values[0]
        max_mean = self.df['mean intensity'].max()
        max_std = self.df[self.df['mean intensity'] == max_mean]['standard deviation'].values[0]
        num_mean = self.df['mean intensity'] - control_mean
        num_std = np.sqrt(self.df['standard deviation'] ** 2 + control_std ** 2)
        den_mean = max_mean - control_mean
        den_std = np.sqrt(max_std ** 2 + control_std ** 2)
        norm_std = np.abs(num_mean / den_mean) * np.sqrt((num_std / num_mean) ** 2 + (den_std / den_mean) ** 2)
        self.df['standard deviation normalized'] = norm_std

    def get_cnr(self):
        control_mean = self.df[self.df['well'] == 'Control']['mean intensity'].values[0]
        control_std = self.df[self.df['well'] == 'Control']['standard deviation'].values[0]
        self.df['CNR'] = (self.df['mean intensity'] - control_mean) / control_std

    def get_ROI_ID(self):
        
        self.df['ROI ID'] = []

    def get_stats(self, region_of_well_to_analyze=0.5):
        # Calculates mean intensity using ROI
        self.df['Analyzed ROI Diameter'] = self.df['ROI Diameter']*region_of_well_to_analyze
        for idx, well in self.df.iterrows():
            if not (np.isfinite(well.x) and np.isfinite(well.y) and well['ROI Diameter'] > 0):
                print(f"Skipping invalid well data at index {idx}")
                continue
            masked_im = self.circular_mask((well.x, well.y), well['Analyzed ROI Diameter']/2)
            self.get_mean_intensity(masked_im, idx)
            self.get_standard_deviation(masked_im, idx)
        
        self.get_mean_intensity_baselined()
        self.get_normalized_column('mean intensity')
        self.get_normalized_std()
        self.get_cnr()

        return self.df
