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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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

from src import SequentialStepsClass, StepOutputsClass

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection

# %% Parameter Optimization Steps
class ParamOptimizer_BIGFISH_SpotDetection_Output(StepOutputsClass):
    def __init__(self, array_of_outputs, order_of_index: dict):
        super().__init__()
        self.array_of_outputs = array_of_outputs
        self.order_of_index = order_of_index

    def append(self, newOutput):
        # for each element in array_of_outputs, append the corresponding element in newOutput.array_of_outputs
        for i, output in np.ndenumerate(self.array_of_outputs):
            output.append(newOutput.array_of_outputs[i]) #  This is awesome cause it mutable 

        

class ParamOptimizer_BIGFISH_SpotDetection(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, id, bigfish_mean_threshold: list[float] = [None], bigfish_alpha: list[float] = [0.7], bigfish_beta: list[float] = [1],
             bigfish_gamma: list[float] = [5], CLUSTER_RADIUS: list[float] = [500],
             MIN_NUM_SPOT_FOR_CLUSTER: list[int] = [4], **kwargs):
        
        self.mean_threshold = bigfish_mean_threshold if type(bigfish_mean_threshold) is list else [bigfish_mean_threshold]
        self.alpha = bigfish_alpha if type(bigfish_alpha) is list else [bigfish_alpha]
        self.beta = bigfish_beta if type(bigfish_beta) is list else [bigfish_beta]
        self.gamma = bigfish_gamma  if type(bigfish_gamma) is list else [bigfish_gamma]
        self.cluster_radius = CLUSTER_RADIUS if type(CLUSTER_RADIUS) is list else [CLUSTER_RADIUS]
        self.min_num_spot_for_cluster = MIN_NUM_SPOT_FOR_CLUSTER if type(MIN_NUM_SPOT_FOR_CLUSTER) is list else [MIN_NUM_SPOT_FOR_CLUSTER]

        order_of_index = {'mean_threshold': 0, 'alpha': 1, 'beta': 2, 'gamma': 3, 'cluster_radius': 4, 'min_num_spot_for_cluster': 5}
        output = np.empty((len(self.mean_threshold), len(self.alpha), len(self.beta), len(self.gamma), len(self.cluster_radius), len(self.min_num_spot_for_cluster)), dtype=object)
        for i, threshold in enumerate(self.mean_threshold):
            for j, alpha in enumerate(self.alpha):
                for k, beta in enumerate(self.beta):
                    for l, gamma in enumerate(self.gamma):
                        for m, radius in enumerate(self.cluster_radius):
                            for n, min_num in enumerate(self.min_num_spot_for_cluster):
                                print(f'Running threshold: {threshold}, alpha: {alpha}, beta: {beta}, gamma: {gamma}, cluster_radius: {radius}, min_num: {min_num}')
                                kwargs['bigfish_mean_threshold'] = threshold
                                kwargs['bigfish_alpha'] = alpha
                                kwargs['bigfish_beta'] = beta
                                kwargs['bigfish_gamma'] = gamma
                                kwargs['CLUSTER_RADIUS'] = radius
                                kwargs['MIN_NUM_SPOT_FOR_CLUSTER'] = min_num
                                kwargs['id'] = id
                                single_output = BIGFISH_SpotDetection().main(**kwargs)
                                if output[i,j,k,l,m,n] is None:
                                    output[i,j,k,l,m,n] = single_output
                                else:
                                    output[i,j,k,l,m,n].append(single_output)

        return ParamOptimizer_BIGFISH_SpotDetection_Output(output, order_of_index)










