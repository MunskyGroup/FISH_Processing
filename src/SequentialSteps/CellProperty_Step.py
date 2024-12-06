import numpy as np
import pathlib
import os
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
import skimage as sk
import dask.array as da
from copy import copy, deepcopy

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection
from src.GeneralOutput import OutputClass
from src.GeneralStep import SequentialStepsClass
import pandas as pd

class CellPropertyOutput(OutputClass):
    def append(self, df):
        if hasattr(self, 'cell_properties'):
            self.cell_properties = pd.concat([self.cell_properties, df], axis=0)
        else:
            self.cell_properties = df

class CellProperties(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, image, nuc_mask, cell_mask, fov, timepoint, 
             props_to_measure= ['label', 'bbox', 'area', 'centroid', 'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std'], **kwargs):
        image = np.max(image.compute(), axis=1)
        image = image.squeeze()
        image = np.moveaxis(image, 0, -1)
    
        nuc_mask = nuc_mask[0, :, :].compute()
        cell_mask = cell_mask[0, :, :].compute()
        plt.imshow(cell_mask)
        plt.show()
        nuc_mask = nuc_mask.squeeze()
        cell_mask = cell_mask.squeeze()

        # make cyto mask
        cyto_mask = copy(cell_mask)
        cyto_mask[nuc_mask > 0] = 0

        plt.imshow(cyto_mask)
        plt.show()

        plt.imshow(cell_mask)
        plt.show()

        nuc_props = sk.measure.regionprops_table(nuc_mask.astype(int), image, properties=props_to_measure)
        cell_props = sk.measure.regionprops_table(cell_mask.astype(int), image, properties=props_to_measure)
        cyto_props = sk.measure.regionprops_table(cyto_mask.astype(int), image, properties=props_to_measure)

        nuc_df = pd.DataFrame(nuc_props)
        cell_df = pd.DataFrame(cell_props)
        cyto_df = pd.DataFrame(cyto_props)

        nuc_df.columns = ['nuc_' + col for col in nuc_df.columns]
        cell_df.columns = ['cell_' + col for col in cell_df.columns]
        cyto_df.columns = ['cyto_' + col for col in cyto_df.columns]

        combined_df = pd.concat([nuc_df, cell_df, cyto_df], axis=1)
        combined_df['fov'] = [fov]*len(combined_df)
        combined_df['timepoint'] = [timepoint]*len(combined_df)

        CellPropertyOutput(combined_df)






























