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
from datetime import datetime

from src.GeneralStep import FinalizingStepClass
from src.Parameters import Parameters
from src.GeneralOutput import OutputClass
from src.Util.Plots import Plots
from src.Util.Metadata import Metadata
from src.Util.ReportPDF import ReportPDF
from src.Util.Utilities import Utilities
from src.Util.NASConnection import NASConnection



class Save_Outputs(FinalizingStepClass):
    def main(self, **kwargs):
        params = Parameters.get_parameters()

        h5_file = params['h5_file']
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']

        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        OutputClass.save_all_outputs(local_dataset_location, h5_file, f'Analysis_{Analysis_name}_{date}')
        





































