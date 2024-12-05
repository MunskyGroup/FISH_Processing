import numpy as np
import pathlib
import os
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
import skimage as sk

from src.Util import Utilities, Plots, CellSegmentation, SpotDetection
from src import StepOutputsClass, SequentialStepsClass


class CellProperties(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, image, nuc_mask, cyto_mask, props_to_measure):
        pass






























