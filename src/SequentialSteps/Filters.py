import numpy as np
from skimage.io import imread
import tifffile
import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from skimage.measure import find_contours
from scipy import signal
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib as mpl
import trackpy as tp
# import bigfish.segmentation as segmentation

mpl.rc('image', cmap='viridis')
plt.style.use('ggplot')  # ggplot  #default
import multiprocessing
from smb.SMBConnection import SMBConnection
import socket
import pathlib
import yaml
import shutil
from fpdf import FPDF
import gc
import pickle
import pycromanager as pycro
import pandas as pd
import cellpose
from cellpose import models


import torch
import warnings
# import tensorflow as tf

import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import bigfish.plot as plot

from typing import Union

warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# Selecting the GPU. This is used in case multiple scripts run in parallel.
try:
    import torch

    number_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    if number_gpus > 1:  # number_gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(np.random.randint(0, number_gpus, 1)[0])
except:
    print('No GPUs are detected on this computer. Please follow the instructions for the correct installation.')
import zipfile
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar

font_props = {'size': 16}
import joypy
from matplotlib import cm
from scipy.ndimage import binary_dilation
import sys
# append the path two directories before this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import SequentialStepsClass, StepOutputsClass, SingleStepCompiler

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection

class filter_output(StepOutputsClass):
    def __init__(self, image: np.array):
        super().__init__()
        self.list_images = [image]

    def append(self, new_output):
        self.list_images = [*self.list_images, *new_output.list_images]


class rescale_images(SequentialStepsClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self, image: np.array, id: int, channel_to_stretch: int = None, stretching_percentile:float = 99.9, display_plots: bool = False, **kwargs):
        # reshape image from zyxc to czyx
        image = np.moveaxis(image, -1, 0)
        # rescale image
        print(image.shape)
        image = stack.rescale(image, channel_to_stretch=channel_to_stretch, stretching_percentile=stretching_percentile)

        # reshape image back to zyxc
        image = np.moveaxis(image, 0, -1)

        if display_plots:
            for c in range(image.shape[3]):
                plt.imshow(np.max(image[:, :, :, c], axis=0))
                plt.title(f'channel {c}')
                plt.show()

        return filter_output(image)

        

class remove_background(SequentialStepsClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self, image: np.array, FISHChannel: list[int], id: int, 
             filter_type: str = 'gaussian', sigma: float = 3, display_plots:bool = False, 
             kernel_shape: str = 'disk', kernel_size = 200, **kwargs):

        rna = np.squeeze(image[:, :, :, FISHChannel[0]])

        if display_plots:
            plt.imshow(np.max(rna, axis=0) if len(rna.shape) > 2 else rna)
            plt.title(f'pre-filtered image')
            plt.show()

        if filter_type == 'gaussian':
            rna = stack.remove_background_gaussian(rna, sigma=sigma)
        elif filter_type == 'log_filter':
            rna = stack.log_filter(rna, sigma=sigma)

        elif filter_type == 'mean':
            rna = stack.remove_background_mean(np.max(rna, axis=0) if len(rna.shape) > 2 else rna, 
                                               kernel_shape=kernel_shape, kernel_size=kernel_size)
        else:
            raise ValueError('Invalid filter type')
        
        image[:, :, :, FISHChannel[0]] = rna

        if display_plots:
            plt.imshow(np.max(rna, axis=0) if len(rna.shape) > 2 else rna)
            plt.title(f'filtered image, type: {filter_type}, sigma: {sigma}')
            plt.show()

        return filter_output(image)




if __name__ == '__main__':
    matplotlib.use('TKAgg')

    ds = pycro.Dataset(r"C:\Users\Jack\Desktop\H128_Tiles_100ms_5mW_Blue_15x15_10z_05step_2")
    kwargs = {'nucChannel': [0], 
              'FISHChannel': [0],
              'user_select_number_of_images_to_run': 5,

              # rescale images
              'channel_to_stretch': 0,
              }
    compiler = SingleStepCompiler(ds, kwargs)
    plt.imshow(np.max(compiler.list_images[0][:, :, :, 0], axis=0))
    plt.title('original image=========')
    plt.show()
    output = compiler.sudo_run_step(rescale_images)
    compiler.list_images = output.list_images
    plt.imshow(np.max(compiler.list_images[0][:, :, :, 0], axis=0))
    plt.title('rescaled image=========')
    plt.show()
    compiler.sudo_run_step(remove_background)
    plt.imshow(np.max(compiler.list_images[0][:, :, :, 0], axis=0)) 
    plt.title('filtered image=========')
    plt.show()