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
class return_to_NAS(Moving_Data):
    def main(self, local_dataset_location, initial_data_location, connection_config_location, share_name, **kwargs):

        for i, h5_file in enumerate(local_dataset_location):
            log_file = os.path.splitext(os.path.basename(h5_file))[0]+'.log'
            with open(log_file, 'a') as log:
                log.write(f'{datetime.now()}: {h5_file} -> {initial_data_location[i]}\n')
            NASConnection(connection_config_location, share_name=share_name).write_files_to_NAS(log_file, '/Users/Jack/All_Analysis')
            NASConnection(connection_config_location, share_name=share_name).write_files_to_NAS(h5_file, initial_data_location[i])

class remove_local_data(Moving_Data):
    def main(self, local_dataset_location, **kwargs):
        for folder in local_dataset_location:
            shutil.rmtree(os.path.dirname(folder))

class remove_local_data_but_keep_h5(Moving_Data):
    def main(self, local_dataset_location, **kwargs):
        for folder in local_dataset_location:
            folder = os.path.abspath(folder)
            for file in os.listdir(os.path.dirname(folder)):
                if file.endswith(".h5"):
                    continue
                else:
                    os.remove(os.path.join(os.path.dirname(folder), file))

class remove_temp(Moving_Data):
    def main(self, temp_name, **kwargs):
        shutil.rmtree(temp_name)

class remove_all_temp(Moving_Data):
    def main(self, temp_name, **kwargs):
        for root, dirs, files in os.walk(temp_name):
            for dir in dirs:
                if 'temp' in dir:
                    shutil.rmtree(os.path.join(root, dir))
        shutil.rmtree(temp_name)



















































