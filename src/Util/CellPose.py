from joblib import Parallel, delayed
import multiprocessing
import bigfish.stack as stack
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import re
from skimage.io import imread
from scipy.optimize import curve_fit
from cellpose import models
import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from skimage.measure import find_contours
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='viridis')
plt.style.use('ggplot')  # ggplot  #default
import multiprocessing
import gc

import warnings

warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# Selecting the GPU. This is used in case multiple scripts run in parallel.
try:
    import torch
    number_gpus = len ( [torch.cuda.device(i) for i in range(torch.cuda.device_count())] )
    if number_gpus >1 : # number_gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] =  str(np.random.randint(0,number_gpus,1)[0])
except:
    print('No GPUs are detected on this computer. Please follow the instructions for the correct installation.')
import seaborn as sns
from  matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
font_props = {'size': 16}

from src.Util.Utilities import Utilities


class Cellpose:
    """
    This class is intended to detect cells by image masking using `Cellpose <https://github.com/MouseLand/cellpose>`_ .
    The class uses optimization to maximize the number of cells or maximize the size of the detected cells.
    For a complete description of Cellpose check the
    `Cellpose documentation <https://cellpose.readthedocs.io/en/latest/>`_ .

    Parameters

    image : NumPy array
        of images with dimensions [Z, Y, X, C].
    num_iterations : int, optional
        Number of iterations for the optimization process. The default is 5.
    channels : List, optional
        List with the channels in the image. For gray images use [0, 0], for RGB images with intensity for cytosol and nuclei use [0, 1] . The default is [0, 0].
    diameter : float, optional
        Average cell size. The default is 120.
    model_type : str, optional
        Cellpose model type the options are 'cyto' for cytosol or 'nuclei' for the nucleus. The default is 'cyto'.
    selection_method : str, optional
        Option to use the optimization algorithm to maximize the number of cells or maximize the size options are 'max_area' or 'max_cells' or 'max_cells_and_area'. The default is 'max_cells_and_area'.
    NUMBER_OF_CORES : int, optional
        The number of CPU cores to use for parallel computing. The default is 1.
    """

    def __init__(self, image: np.ndarray, num_iterations: int = 6, channels: list = [0, 0], diameter: float = 120,
                 model_type: str = 'cyto', selection_method: str = 'cellpose_max_cells_and_area',
                 NUMBER_OF_CORES: int = 1, pretrained_model=None):
        self.image = image
        self.num_iterations = num_iterations
        self.minimum_flow_threshold = 0.1
        self.maximum_flow_threshold = 0.9
        self.channels = channels
        self.diameter = diameter
        self.model_type = model_type  # options are 'cyto' or 'nuclei'
        self.selection_method = selection_method  # options are 'max_area' or 'max_cells'
        self.NUMBER_OF_CORES = NUMBER_OF_CORES
        self.default_flow_threshold = 0.4  # default is 0.4
        self.optimization_parameter = np.unique(
            np.round(np.linspace(self.minimum_flow_threshold, self.maximum_flow_threshold, self.num_iterations), 2))
        self.MINIMUM_CELL_AREA = np.pi * (diameter / 4) ** 2  # 1000  # using half of the diameter to calculate area.
        self.BATCH_SIZE = 80
        self.pretrained_model = pretrained_model

    def calculate_masks(self):
        """
        This method performs the process of image masking using **Cellpose**.

        Returns

        selected_masks : List of NumPy arrays
            List of NumPy arrays with values between 0 and the number of detected cells in the image, where an integer larger than zero represents the masked area for each cell, and 0 represents the background in the image.
        """
        # Next two lines suppressing output from Cellpose
        gc.collect()
        torch.cuda.empty_cache()
        if self.pretrained_model is None:
            self.model = models.Cellpose(gpu=1, model_type=self.model_type)  # model_type = 'cyto' or model_type = 'nuclei'
        else:
            self.model = models.CellposeModel(gpu=1,
                                         pretrained_model=self.pretrained_model)  # model_type = 'cyto' or model_type = 'nuclei'

        # Loop that test multiple probabilities in cell pose and returns the masks with the longest area.


        if self.selection_method == 'max_area':
            list_metrics_masks = Parallel(n_jobs=self.NUMBER_OF_CORES)(
                delayed(self.cellpose_max_area)(tested_parameter) for _, tested_parameter in
                enumerate(self.optimization_parameter))
            evaluated_metric_for_masks = np.array(list_metrics_masks)
        if self.selection_method == 'max_cells':
            list_metrics_masks = Parallel(n_jobs=self.NUMBER_OF_CORES)(
                delayed(self.cellpose_max_cells)(tested_parameter) for _, tested_parameter in
                enumerate(self.optimization_parameter))
            evaluated_metric_for_masks = np.array(list_metrics_masks)
        if self.selection_method == 'max_cells_and_area':
            # list_metrics_masks = [self.cellpose_max_cells_and_area(i) for i in self.optimization_parameter]
            list_metrics_masks = Parallel(n_jobs=self.NUMBER_OF_CORES)(
                delayed(self.cellpose_max_cells_and_area)(tested_parameter) for _, tested_parameter in
                enumerate(self.optimization_parameter))
            evaluated_metric_for_masks = np.array(list_metrics_masks)
        # if not (self.selection_method is None) and (np.max(evaluated_metric_for_masks) > 0):
        # if np.max(evaluated_metric_for_masks) > 0:
        #     print('idk why the evaluation criteria is shit (Cellpose)')
        if (self.selection_method is not None) and (np.max(evaluated_metric_for_masks) > 0):
            selected_conditions = self.optimization_parameter[np.argmax(evaluated_metric_for_masks)]
            selected_masks = \
            self.model.eval(self.image, batch_size=self.BATCH_SIZE, normalize=True, flow_threshold=selected_conditions,
                       diameter=self.diameter, min_size=self.MINIMUM_CELL_AREA, channels=self.channels, progress=None)[
                0]
            selected_masks = Utilities().remove_artifacts_from_mask_image(selected_masks,
                                                                          minimal_mask_area_size=self.MINIMUM_CELL_AREA)
        else:
            if len(self.image.shape) >= 3:
                selected_masks = np.zeros_like(self.image[:, :, 0])
            else:
                selected_masks = np.zeros_like(self.image[:, :])
        # If no GPU is available, the segmentation is performed with a single threshold.
        if self.selection_method is None:
            selected_masks = self.model.eval(self.image, batch_size=self.BATCH_SIZE, normalize=True,
                                        flow_threshold=self.default_flow_threshold, diameter=self.diameter,
                                        min_size=self.MINIMUM_CELL_AREA, channels=self.channels, progress=None)[0]
            selected_masks = Utilities().remove_artifacts_from_mask_image(selected_masks,
                                                                          minimal_mask_area_size=self.MINIMUM_CELL_AREA)
        return selected_masks

    def cellpose_max_area(self, optimization_parameter):
        try:
            masks = self.model.eval(self.image, batch_size=self.BATCH_SIZE, normalize=True,
                               flow_threshold=optimization_parameter, diameter=self.diameter,
                               min_size=self.MINIMUM_CELL_AREA, channels=self.channels, progress=None)[0]
            # removing artifacts.
            masks = Utilities().remove_artifacts_from_mask_image(masks,
                                                                 minimal_mask_area_size=self.MINIMUM_CELL_AREA)
        except:
            masks = 0
        n_masks = np.max(masks)
        if n_masks > 1:  # detecting if more than 1 mask are detected per cell
            size_mask = []
            for nm in range(1,
                            n_masks + 1):  # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                size_mask.append(np.sum(masks == nm))  # creating a list with the size of each mask
            largest_mask = np.argmax(size_mask) + 1  # detecting the mask with the largest value
            temp_mask = np.zeros_like(masks)  # making a copy of the image
            selected_mask = temp_mask + (
                        masks == largest_mask)  # Selecting a single mask and making this mask equal to one and the background equal to zero.
            return np.sum(selected_mask)
        else:  # do nothing if only a single mask is detected per image.
            return np.sum(masks)

    def cellpose_max_cells(self, optimization_parameter):
        try:
            masks = self.model.eval(self.image, batch_size=self.BATCH_SIZE, normalize=True,
                               flow_threshold=optimization_parameter, diameter=self.diameter,
                               min_size=self.MINIMUM_CELL_AREA, channels=self.channels, progress=None)[0]
            # removing artifacts.
            masks = Utilities().remove_artifacts_from_mask_image(masks,
                                                                 minimal_mask_area_size=self.MINIMUM_CELL_AREA)
        except:
            masks = 0
        return np.max(masks)

    def cellpose_max_cells_and_area(self, optimization_parameter):
        try:
            masks = self.model.eval(self.image, batch_size=self.BATCH_SIZE, normalize=True,
                               flow_threshold=optimization_parameter, diameter=self.diameter,
                               min_size=self.MINIMUM_CELL_AREA, channels=self.channels, progress=None)[0]
            # removing artifacts.
            masks = Utilities().remove_artifacts_from_mask_image(masks,
                                                                 minimal_mask_area_size=self.MINIMUM_CELL_AREA)
        except:
            print('Bull shit is happening: 111')
            masks = 0
        print(masks)
        number_masks = np.max(masks)
        if number_masks > 1:  # detecting if more than 1 mask are detected per cell
            size_mask = []
            for nm in range(1, number_masks + 1):
                # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                approximated_radius = np.sqrt(np.sum(masks == nm) / np.pi)  # a=  pi r2
                size_mask.append(
                    approximated_radius)  # np.sum(masks == nm)) # creating a list with the size of each mask
            size_masks_array = np.array(size_mask)
            metric = np.mean(size_masks_array).astype(int) * number_masks
        elif number_masks == 1:  # do nothing if only a single mask is detected per image.
            approximated_radius = np.sqrt(np.sum(masks == 1) / np.pi)
            metric = approximated_radius.astype(int)
        else:  # return zero if no mask are detected
            metric = 0
        return metric
