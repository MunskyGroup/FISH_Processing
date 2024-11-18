import pathlib
import numpy as np
import shutil
from fpdf import FPDF
import os
import pickle
import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import dask.array as da
from datetime import datetime
from abc import abstractmethod

from src.GeneralStep import FinalizingStepClass
from src.Parameters import Parameters
from src.GeneralOutput import OutputClass

from src.Util.NASConnection import NASConnection



#%% abstract class for moving data
class Moving_Data(FinalizingStepClass):
    @abstractmethod
    def main(self, **kwargs):
        pass


#%% class for moving data to NAS
class return_to_NAS(FinalizingStepClass):
    def main(self, local_dataset_location, initial_data_location, connection_config_location, share_name, **kwargs):
        # shutil.make_archive(analysis_location,'zip', pathlib.Path().absolute().joinpath(analysis_location))
        NASConnection(connection_config_location,share_name = share_name).write_files_to_NAS(local_dataset_location, initial_data_location)















































