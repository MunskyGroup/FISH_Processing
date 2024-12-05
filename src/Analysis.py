import h5py
import matplotlib.pyplot as plt
import dask.array as da
import numpy as np
import dask.dataframe as dp
import pandas as pd
from typing import Union

"""
self: for each instance of this class has its own self.
    contains:
        methods
        variables
"""

class Analysis:
    def __init__(self, location: Union[str, list[str]] = None):
        # given:
        # h5 locations
        #   give me a location
        #   give me a list of locations
        #   give me none -> got to here and display these \\munsky-nas.engr.colostate.edu\share\Users\Jack\All_Analysis
        if location is None:
            self.select_from_list()
        elif isinstance(location, str):
            self.location = [location]
        elif isinstance(location, list): # TODO make sure its a list of str
            self.location = location
        else:
            raise ValueError('Location is not properly defined')
        
        self._load_in_h5()
        
    def select_from_list(self) -> list[str]: # TODO: this requires user input
        pass

    def select_analysis(self, analysis_name: str = None, date_range: list[str] = None):
        self._find_analysis_names()
        self._filter_on_date(date_range)
        self._filter_on_name(analysis_name)
        self._find_analysis()
        self._deal_with_duplicates()

    def list_analysis_names(self):
        self._find_analysis_names()
        for name in self.analysis_names:
            print(name)
        return self.analysis_names

    def select_datasets(self, dataset_name):
        self.datasets = [h[dataset_name] for h in self.analysis]
        return self.datasets

    def list_datasets(self):
        for d in self.analysis:
            print(d.name, list(d.keys()))

    def _filter_on_name(self, analysis_name):
        self.analysis_names = [s.split('_')[1] for s in self.analysis_names]
        if analysis_name is not None:
            self.analysis_names = [s for s in self.analysis_names if s == analysis_name]

    def _filter_on_date(self, date_range):
        self.dates = set([s.split('_')[2] for s in self.analysis_names])
        if date_range is not None:
            start_date, end_date = date_range
            self.dates = [date for date in self.dates if start_date <= date <= end_date]
        self.dates = list(self.dates)

    def _find_analysis(self):
        # select data sets with self.data, and self.datasete
        self.analysis = []
        for h in self.h5_files:
            for dataset_name in self.analysis_names:
                for date in self.dates:
                    if f'Analysis_{dataset_name}_{date}' in list(h.keys()):
                        self.analysis.append(h[f'Analysis_{dataset_name}_{date}'])

    def _deal_with_duplicates(self): # requires user input
        pass

    def _find_analysis_names(self):
        self.analysis_names = []
        for h in self.h5_files:
            self.analysis_names.append(list(h.keys()))
        self.analysis_names = set([dataset for sublist in self.analysis_names for dataset in sublist])
        self.analysis_names = [d for d in self.analysis_names if 'Analysis' in d]

    def _load_in_h5(self):
        self.h5_files = []
        for l in self.location:
            self.h5_files.append(h5py.File(l, 'r'))














if __name__ == '__main__':
    ana = Analysis(r'\\munsky-nas.engr.colostate.edu\share\smFISH_images\Eric_smFISH_images\20220225\DUSP1_Dex_0min_20220224\DUSP1_Dex_0min_20220224.h5')
    print(ana.location)
    print(ana.h5_files)
    ana.list_analysis_names()
    ana.select_analysis()
    ana.list_datasets()
    print(ana.select_datasets('df_spotresults'))


