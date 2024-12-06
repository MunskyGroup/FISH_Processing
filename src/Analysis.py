import h5py
import matplotlib.pyplot as plt
import dask.array as da
import numpy as np
import dask.dataframe as dp
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod

"""
self: for each instance of this class has its own self.
    contains:
        methods
        variables
"""

class Analysis_Manager:
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

    def select_datasets(self, dataset_name) -> list:
        if hasattr(self, 'analysis'):
            self.datasets = [h[dataset_name] for h in self.analysis]
            return self.datasets
        else:
            print('select an anlysis')

    def list_datasets(self):
        if hasattr(self, 'analysis'):
            for d in self.analysis:
                print(d.name, list(d.keys()))
        else:
            print('select an analysis')

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

        self.raw_images = [da.from_array(h['raw_images']) for h in self.h5_files]
        self.masks = [da.from_array(h['masks']) for h in self.h5_files]

#%%
class Analysis(ABC):
    def __init__(self, am, seed: float = None):
        super().__init__()
        self.am = am
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def save_data(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def select_data(self, identifying_feature):
        pass

class SpotDetection_confirmation(Analysis):
    def get_data(self):
        self.spots = self.am.select_datasets('df_spotresults')
        for i, s in enumerate(self.spots):
            self.spots[i] = pd.read_hdf(s)
            self.spots[i]['h5_idx'] = [i]*len(self.spots[i])
        self.spots = pd.concat(self.spots, axis=0)
        self.images = self.am.raw_images
        self.masks = self.am.masks

    def save_data(self, location):
        self.spots.to_csv(location, index=False)

    def display(self):
        fig, axs = plt.subplot(1, 3)
        # select one spot in the image and draw an arrow pointing to it with a specific color

        # axs[0] - Plot the show image with arrows point at all the spots, with the selected spot being a different color
        # add transperent outlines to the outside of cells

        # axs[1] - Select the cell with the spot in it and show a bound box around that cell
        # put transperent outline around the cell

        # axs[2] - zoom in further on the spot and show a nxn box around the spot
        spot = self.spots.sample(n=1, random_state=self.seed).iloc[0]
        h5_idx = spot['h5_idx']
        image = self.images[h5_idx]
        mask = self.masks[h5_idx]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # axs[0] - Plot the show image with arrows pointing at all the spots, with the selected spot being a different color
        axs[0].imshow(image, cmap='gray')
        for _, s in self.spots.iterrows():
            color = 'red' if s.equals(spot) else 'blue'
            axs[0].arrow(s['x'], s['y'], 0, 0, color=color, head_width=5)
        axs[0].set_title('All spots with selected spot highlighted')

        # axs[1] - Select the cell with the spot in it and show a bound box around that cell
        cell_mask = mask == mask[int(spot['y']), int(spot['x'])]
        axs[1].imshow(image, cmap='gray')
        axs[1].imshow(cell_mask, cmap='jet', alpha=0.5)
        axs[1].set_title('Cell containing the selected spot')

        # axs[2] - Zoom in further on the spot and show a nxn box around the spot
        n = 20
        x, y = int(spot['x']), int(spot['y'])
        zoomed_image = image[max(0, y-n):y+n, max(0, x-n):x+n]
        axs[2].imshow(zoomed_image, cmap='gray')
        axs[2].set_title('Zoomed in on selected spot')

        plt.show()








if __name__ == '__main__':
    ana = Analysis_Manager(r'\\munsky-nas.engr.colostate.edu\share\smFISH_images\Eric_smFISH_images\20220225\DUSP1_Dex_0min_20220224\DUSP1_Dex_0min_20220224.h5')
    print(ana.location)
    print(ana.h5_files)
    ana.list_analysis_names()
    ana.select_analysis()
    ana.list_datasets()
    print(ana.select_datasets('df_spotresults'))


