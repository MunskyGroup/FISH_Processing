from ufish.api import UFish
import matplotlib.pyplot as plt
from pycromanager import Dataset
import numpy as np
import os
import sys
import pandas as pd
from typing import Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pathlib
from bigfish import stack, detection, multistack, plot
import trackpy as tp
import tifffile
from abc import abstractmethod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.GeneralOutput import OutputClass
from src import SequentialStepsClass, IndependentStepClass
from src.Util import Plots, SpotDetection
from src.Parameters import Parameters


#%% Output Classes
class SpotDetectionOutputClass(OutputClass):
    def append(self, df_cellresults, df_spotresults, df_clusterresults, threshold):
        if not hasattr(self, 'df_cellresults'):
            self.df_cellresults = df_cellresults
        if self.df_cellresults is not None:
            self.df_cellresults = pd.concat([self.df_cellresults, df_cellresults])
        
        if not hasattr(self, 'df_spotresults'):
            self.df_spotresults = df_spotresults
        if self.df_spotresults is not None:
            self.df_spotresults = pd.concat([self.df_spotresults, df_spotresults])
        
        if not hasattr(self, 'df_clusterresults'):
            self.df_clusterresults = df_clusterresults
        if self.df_clusterresults is not None:
            self.df_clusterresults = pd.concat([self.df_clusterresults, df_clusterresults])

        if not hasattr(self, 'bigfish_threshold'):
            self.bigfish_threshold = [threshold]
        if threshold is not None:
            self.bigfish_threshold = [*self.bigfish_threshold, threshold]

class New_Parameters(OutputClass):
    def append(self, new_params):
        Parameters.update_parameters(new_params)

#%% Abstract Class
class SpotDetection(SequentialStepsClass):
    def main(self, image, nuc_mask, cell_mask, nucChannel, cytoChannel, FISHChannel, timepoint, fov, verbose, display_plots,
             voxel_size_yx, voxel_size_z, spot_yx, spot_z, **kwargs) -> SpotDetectionOutputClass:
        for c in range(len(FISHChannel)):
            rna = image[FISHChannel[c], :, :, :]
            rna = rna.squeeze()
            rna = rna.compute()
            spots, clusters = self.get_detected_spots(**kwargs)

            cell_results = self.extract_cell_level_results(image, spots, clusters, nucChannel, c, 
                                                        nuc_mask, cell_mask, timepoint, fov,
                                                            verbose, display_plots)

            spots, clusters = self.get_spot_properties(rna, spots, voxel_size_yx, voxel_size_z, spot_yx, spot_z, display_plots, **kwargs)

            spots, cell_results, clusters = self.add_ind_params(spots, cell_results, clusters, timepoint, fov, c)

        return SpotDetectionOutputClass(cell_results, spots, clusters)


    @abstractmethod
    def get_detected_spots(self, **kwargs) -> pd.DataFrame:
        pass

    def get_spot_properties(self, rna, spots, voxel_size_yx, voxel_size_z, spot_yx, spot_z, display_plots, **kwargs) -> pd.DataFrame:
        voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
        spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))
        snr, signal = compute_snr_spots(rna, spots[:, :len(voxel_size_nm)], voxel_size_nm, spot_size_nm, display_plots)
        snr = np.array(snr).reshape(-1, 1)
        signal = np.array(signal).reshape(-1, 1)
        spots = np.hstack([spots, snr, signal])
        return spots

    def extract_cell_level_results(self, image, spots, clusters, nucChannel, FISHChannel, 
                                   nuc_mask, cell_mask, timepoint, fov,
                                    verbose, display_plots) -> pd.DataFrame:
        if (nuc_mask is not None and nuc_mask.max() != 0 or cell_mask is not None and cell_mask.max() != 0):

            nuc = image[nucChannel, :, :, :].squeeze().compute()
            rna = image[FISHChannel, :, :, :].squeeze().compute()

            # convert masks to max projection
            if nuc_mask is not None and len(nuc_mask.shape) != 2:
                nuc_mask = np.max(nuc_mask, axis=0)
            if cell_mask is not None and len(cell_mask.shape) != 2:
                cell_mask = np.max(cell_mask, axis=0)

            ndim = 2
            if len(rna.shape) == 3:
                ndim = 3
                rna = np.max(rna, axis=0)

            # convert types
            nuc_mask = nuc_mask.squeeze().astype("uint16").compute() if nuc_mask is not None else None
            cell_mask = cell_mask.squeeze().astype("uint16").compute() if cell_mask is not None else None

            # remove transcription sites
            spots_no_ts, foci, ts = multistack.remove_transcription_site(spots, clusters, nuc_mask, ndim=3)
            if verbose:
                print("detected spots (without transcription sites)")
                print("\r shape: {0}".format(spots_no_ts.shape))
                print("\r dtype: {0}".format(spots_no_ts.dtype))

            # get spots inside and outside nuclei
            spots_in, spots_out = multistack.identify_objects_in_region(nuc_mask, spots, ndim=3)
            if verbose:
                print("detected spots (inside nuclei)")
                print("\r shape: {0}".format(spots_in.shape))
                print("\r dtype: {0}".format(spots_in.dtype), "\n")
                print("detected spots (outside nuclei)")
                print("\r shape: {0}".format(spots_out.shape))
                print("\r dtype: {0}".format(spots_out.dtype))

            # extract fov results
            cell_mask = cell_mask.astype("uint16") if cell_mask is not None else nuc_mask.astype("uint16")
            nuc_mask = nuc_mask.astype("uint16") if nuc_mask is not None else None
            rna = rna.astype("uint16")
            other_images = {}
            other_images["dapi"] = np.max(nuc, axis=0).astype("uint16") if nuc is not None else None

            fov_results = multistack.extract_cell( # this function is incredibly poorly written be careful looking at it
                cell_label=cell_mask,
                ndim=ndim,
                nuc_label=nuc_mask,
                rna_coord=spots_no_ts,
                others_coord={"foci": foci, "transcription_site": ts},
                image=rna,
                others_image=other_images,)
            if verbose:
                print("number of cells identified: {0}".format(len(fov_results)))

            # cycle through cells and save the results
            for i, cell_results in enumerate(fov_results):
                # get cell results
                cell_mask = cell_results["cell_mask"]
                cell_coord = cell_results["cell_coord"]
                nuc_mask = cell_results["nuc_mask"]
                nuc_coord = cell_results["nuc_coord"]
                rna_coord = cell_results["rna_coord"]
                foci_coord = cell_results["foci"]
                ts_coord = cell_results["transcription_site"]
                image_contrasted = cell_results["image"]
                
                if verbose:
                    print("cell {0}".format(i))
                    print("\r number of rna {0}".format(len(rna_coord)))
                    print("\r number of foci {0}".format(len(foci_coord)))
                    print("\r number of transcription sites {0}".format(len(ts_coord)))

                # plot individual cells
                if display_plots:
                    plot.plot_cell(
                        ndim=3, cell_coord=cell_coord, nuc_coord=nuc_coord,
                        rna_coord=rna_coord, foci_coord=foci_coord, other_coord=ts_coord,
                        image=image_contrasted, cell_mask=cell_mask, nuc_mask=nuc_mask, rescale=True, contrast=True,
                        title="Cell {0}".format(i))

            df = multistack.summarize_extraction_results(fov_results, ndim=3)
            df['fov'] = [fov]*len(df)
            df['timepoint'] = [timepoint]*len(df)
            df['FISH_Channel'] = [FISHChannel]*len(df)

        else:
            df = None

        return df

    def add_ind_params(self, df_spotresults, df_cellresults, df_clusterresults, timepoint, fov, c):
        df_spotresults['timepoint'] = [timepoint]*len(df_spotresults)
        df_spotresults['fov'] = [fov]*len(df_spotresults)
        df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

        if df_cellresults is not None:
            df_cellresults['timepoint'] = [timepoint]*len(df_cellresults)
            df_cellresults['fov'] = [fov]*len(df_cellresults)
            df_cellresults['FISH_Channel'] = [c]*len(df_cellresults)

        df_clusterresults['timepoint'] = [timepoint]*len(df_clusterresults)
        df_clusterresults['fov'] = [fov]*len(df_clusterresults)
        df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)

        return df_spotresults, df_cellresults, df_clusterresults



#%% Useful Functions
def compute_snr_spots(image, spots, voxel_size, spot_radius, display_plots: bool = False):
    """Compute signal-to-noise ratio (SNR) based on spot coordinates.

    .. math::

        \\mbox{SNR} = \\frac{\\mbox{max(spot signal)} -
        \\mbox{mean(background)}}{\\mbox{std(background)}}

    Background is a region twice larger surrounding the spot region. Only the
    y and x dimensions are taking into account to compute the SNR.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray
        Coordinate of the spots, with shape (nb_spots, 3) or (nb_spots, 2).
        One coordinate per dimension (zyx or yx coordinates).
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.

    Returns
    -------
    snr : float
        Median signal-to-noise ratio computed for every spots.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_range_value(image, min_=0)
    stack.check_array(
        spots,
        ndim=2,
        dtype=[np.float32, np.float64, np.int32, np.int64])
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list))

    # check consistency between parameters
    ndim = image.ndim
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(spot_radius, (tuple, list)):
        if len(spot_radius) != ndim:
            raise ValueError(
                "'spot_radius' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        spot_radius = (spot_radius,) * ndim

    # cast spots coordinates if needed
    if spots.dtype == np.float64:
        spots = np.round(spots).astype(np.int64)

    # cast image if needed
    image_to_process = image.copy().astype(np.float64)

    # clip coordinate if needed
    if ndim == 3:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)
        spots[:, 2] = np.clip(spots[:, 2], 0, image_to_process.shape[2] - 1)
    else:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)

    # compute radius used to crop spot image
    radius_pixel = detection.get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=spot_radius,
        ndim=ndim)
    radius_signal_ = [np.sqrt(ndim) * r for r in radius_pixel]
    radius_signal_ = tuple(radius_signal_)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * 2 for i in radius_signal_)

    # ceil radii
    radius_signal = np.ceil(radius_signal_).astype(np.uint16)
    radius_background = np.ceil(radius_background_).astype(np.uint16)

    snr_spots = []
    max_signals = []

    # Loop over each spot
    for spot in spots:
        # Extract spot coordinates
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx

        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            # Compute max signal for the current spot
            max_signal = image_to_process[spot_z, spot_y, spot_x]
            # Extract background region
            spot_background_, _ = detection.get_spot_volume(
                image_to_process, spot_z, spot_y, spot_x,
                radius_background_z, radius_background_yx)
            spot_background = spot_background_.copy()

            # Remove signal region from the background crop
            spot_background[:, edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        else:
            # For 2D images
            max_signal = image_to_process[spot_y, spot_x]
            spot_background_, _ = detection.get_spot_surface(
                image_to_process, spot_y, spot_x, radius_background_yx)
            spot_background = spot_background_.copy()

            # Remove signal region from the background crop
            spot_background[edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        # Compute mean and standard deviation of the background
        mean_background = np.mean(spot_background)
        std_background = np.std(spot_background)

        # Compute SNR
        snr = (max_signal - mean_background) / (std_background + 1e-8)  # Add small value to avoid division by zero
        snr_spots.append(snr)
        max_signals.append(max_signal)

    # Return both SNRs and max_signals for further analysis
    return snr_spots, max_signals


#%% Step Classes
class BIGFISH_SpotDetection(SpotDetection):
    """
    A class for detecting RNA spots in FISH images using the BIGFISH library.
    Methods
    -------
    __init__():
        Initializes the BIGFISH_SpotDetection class.
    main(image, FISHChannel, nucChannel, nuc_mask, cell_mask, voxel_size_yx, voxel_size_z, spot_yx, spot_z, timepoint, fov, independent_params, bigfish_threshold=None, snr_threshold=None, snr_ratio=None, bigfish_alpha=0.7, bigfish_beta=1, bigfish_gamma=5, CLUSTER_RADIUS=500, MIN_NUM_SPOT_FOR_CLUSTER=4, use_log_hook=False, verbose=False, display_plots=False, bigfish_use_pca=False, sub_pixel_fitting=False, bigfish_minDistance=None, **kwargs):
        Main method to detect spots in FISH images and extract cell-level results.
        Parameters:
        - image (np.array): Input image.
        - FISHChannel (list): List of FISH channels.
        - nucChannel (list): List of nuclear channels.
        - nuc_mask (np.array): Nuclear mask.
        - cell_mask (np.array): Cell mask.
        - voxel_size_yx (float): Voxel size in the yx plane.
        - voxel_size_z (float): Voxel size in the z plane.
        - spot_yx (float): Spot size in the yx plane.
        - spot_z (float): Spot size in the z plane.
        - timepoint (int): Timepoint of the image.
        - fov (int): Field of view of the image.
        - independent_params (dict): Independent parameters.
        - bigfish_threshold (Union[int, str], optional): Threshold for spot detection.
                            mean, min, max, median, mode, 75th_percentile, 25th_percentile, 90th_percentile.
        - snr_threshold (float, optional): SNR threshold for spot filtering.
        - snr_ratio (float, optional): Ratio to determine SNR threshold.
        - bigfish_alpha (float, optional): Alpha parameter for spot decomposition.
        - bigfish_beta (float, optional): Beta parameter for spot decomposition.
        - bigfish_gamma (float, optional): Gamma parameter for spot decomposition.
        - CLUSTER_RADIUS (float, optional): Radius for clustering spots.
        - MIN_NUM_SPOT_FOR_CLUSTER (int, optional): Minimum number of spots for clustering.
        - use_log_hook (bool, optional): Whether to use log kernel for spot detection.
        - verbose (bool, optional): Whether to print verbose output.
        - display_plots (bool, optional): Whether to display plots.
        - bigfish_use_pca (bool, optional): Whether to use PCA for spot filtering.
        - sub_pixel_fitting (bool, optional): Whether to use sub-pixel fitting for spot detection.
        - bigfish_minDistance (Union[float, list], optional): Minimum distance for spot detection.
    """
    def __init__(self):
        super().__init__()

    def main(self, image, FISHChannel,  nucChannel, nuc_mask, cell_mask,
             voxel_size_yx: int, voxel_size_z: int, spot_yx: int, spot_z: int, timepoint, fov, independent_params: dict,
             bigfish_threshold: Union[int, str] = None, snr_threshold: float = None, snr_ratio: float = None,
             bigfish_alpha: float = 0.7, bigfish_beta:float = 1, bigfish_gamma:float = 5, 
             CLUSTER_RADIUS:int = 500, MIN_NUM_SPOT_FOR_CLUSTER:int = 4, use_log_hook:bool = False, 
             verbose:bool = False, display_plots: bool = False, bigfish_use_pca: bool = False,
             sub_pixel_fitting: bool = False, bigfish_minDistance:Union[float, list] = None, **kwargs):
        
        # cycle through FISH channels
        for c in range(len(FISHChannel)):
            rna = image[FISHChannel[c], :, :, :]
            rna = rna.squeeze()
            rna = rna.compute()

            # detect spots
            spots_px, dense_regions, reference_spot, clusters, spots_subpx, threshold = self.get_detected_spots( FISHChannel=c,
                rna=rna, voxel_size_yx=voxel_size_yx, voxel_size_z=voxel_size_z, spot_yx=spot_yx, spot_z=spot_z, alpha=bigfish_alpha,
                beta=bigfish_beta, gamma=bigfish_gamma, CLUSTER_RADIUS=CLUSTER_RADIUS, MIN_NUM_SPOT_FOR_CLUSTER=MIN_NUM_SPOT_FOR_CLUSTER, 
                bigfish_threshold=bigfish_threshold, use_log_hook=use_log_hook, verbose=verbose, display_plots=display_plots, sub_pixel_fitting=sub_pixel_fitting,
                minimum_distance=bigfish_minDistance, use_pca=bigfish_use_pca, snr_threshold=snr_threshold, snr_ratio=snr_ratio, **kwargs)
            
            cell_results = self.extract_cell_level_results(image, spots_px, clusters, nucChannel, c, 
                                                            nuc_mask, cell_mask, timepoint, fov,
                                                            verbose, display_plots)

            spots_px = self.get_spot_properties(rna, spots_px, voxel_size_yx, voxel_size_z, spot_yx, spot_z, display_plots, **kwargs)

            spots, clusters = self.standardize_df(cell_results, spots_px, spots_subpx, sub_pixel_fitting, clusters, c, timepoint, fov, independent_params)

            output = SpotDetectionOutputClass(cell_results, spots, clusters, threshold)
        return output
        
    def _establish_threshold(self, c, bigfish_threshold, kwargs):
            # check if a threshold is provided
            if type(bigfish_threshold) == int:
                threshold = bigfish_threshold
            elif bigfish_threshold == 'mean':
                threshold = kwargs['bigfish_mean_threshold'][c]
            elif bigfish_threshold == 'min':
                threshold = kwargs['bigfish_min_threshold'][c]
            elif bigfish_threshold == 'max':
                threshold = kwargs['bigfish_max_threshold'][c]
            elif bigfish_threshold == 'median':
                threshold = kwargs['bigfish_median_threshold'][c]
            elif bigfish_threshold == 'mode':
                threshold = kwargs['bigfish_mode_threshold'][c]
            elif bigfish_threshold == '75th_percentile':
                threshold = kwargs['bigfish_75_quartile'][c]
            elif bigfish_threshold == '25th_percentile':
                threshold = kwargs['bigfish_25_quartile'][c]
            elif bigfish_threshold == '90th_percentile':
                threshold = kwargs['bigfish_90_quartile'][c]
            else:
                threshold = None

            return threshold

    def get_detected_spots(self, FISHChannel: int, rna:np.array, voxel_size_yx:float, voxel_size_z:float, spot_yx:float, spot_z:float, alpha:int, beta:int,
                               gamma:int, CLUSTER_RADIUS:float, MIN_NUM_SPOT_FOR_CLUSTER:int, use_log_hook:bool, 
                               verbose: bool = False, display_plots: bool = False, sub_pixel_fitting: bool = False, minimum_distance:Union[list, float] = None,
                               use_pca: bool = False, snr_threshold: float = None, snr_ratio: float = None, bigfish_threshold: Union[int, str] = None,  **kwargs):

        threshold = self._establish_threshold(FISHChannel, bigfish_threshold, kwargs)


        voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
        spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))

        if use_log_hook:
            spot_radius_px = detection.get_object_radius_pixel(
                    voxel_size_nm=voxel_size_nm, 
                    object_radius_nm=spot_size_nm, 
                    ndim=len(rna.shape))
        else:
            spot_radius_px = None

        canidate_spots, individual_thershold = detection.detect_spots(
                                        images=rna, 
                                        return_threshold=True, 
                                        threshold=threshold,
                                        voxel_size=voxel_size_nm if not use_log_hook else None,
                                        spot_radius=spot_size_nm if not use_log_hook else None,
                                        log_kernel_size=spot_radius_px if use_log_hook else None,
                                        minimum_distance=minimum_distance if use_log_hook and minimum_distance is None else spot_radius_px,)
        
        if use_pca:
            # lets try log filter
            log = stack.log_filter(rna.copy(), 3)

            # TODO: PCA for overdetected spots
            print(canidate_spots.shape)
            valid_spots = np.ones(canidate_spots.shape[0])
            canidate_spots = np.array(canidate_spots)
            pca_data = np.zeros((canidate_spots.shape[0], 5*5))

            for i in range(canidate_spots.shape[0]-1):
                xyz = canidate_spots[i, :] # idk why this is being mean to me 
                if len(rna.shape) == 3:
                    x, y, z = xyz
                    try:
                        spot_kernel = log[z-2:z+3, y-2:y+3, x-2:x+3]
                        pca_data[i, :] = spot_kernel.flatten()
                        plt.imshow(spot_kernel)
                    except:
                        valid_spots[i] = 0
                else:
                    x, y = xyz
                    try:
                        spot_kernel = log[y-2:y+3, x-2:x+3]
                        pca_data[i, :] = spot_kernel.flatten()
                        plt.imshow(spot_kernel)
                    except:
                        valid_spots[i] = 0

            plt.show()

            # z score normalization
            pca_data = (pca_data - np.mean(pca_data, axis=0)) / np.std(pca_data, axis=0)
            pca = PCA(n_components=9)
            pca.fit(pca_data)
            X = pca.transform(pca_data)

            # color the spots best on the clusters
            kmeans_pca = KMeans(n_clusters=2)
            kmeans_pca.fit(X)
            plt.scatter(X[:, 0], X[:, 1], c=kmeans_pca.labels_)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.show()

            # remove larger varying clusters
            valid_spots[kmeans_pca.labels_ != 1] = 0
            valid_spots = valid_spots.astype(bool)
            canidate_spots = canidate_spots[valid_spots, :]

            print(canidate_spots.shape)

        # Compute SNR of the detected spots (2D array of coordinates)
        snr_spots, max_signal = compute_snr_spots(
            image=rna.astype(np.float64), 
            spots=canidate_spots.astype(np.float64), 
            voxel_size=voxel_size_nm, 
            spot_radius=spot_size_nm if not use_log_hook else spot_radius_px)
        if display_plots:
            plt.hist(snr_spots, bins=100)
            plt.xlabel('SNR')
            plt.ylabel('Frequency')
            plt.title('SNR distribution')
            plt.show()
        print(f'median SNR: {np.median(snr_spots)}')
        print(f'mean SNR: {np.mean(snr_spots)}')
        if display_plots:
            plt.scatter(max_signal, snr_spots, color='blue', alpha=0.5, s=3)
            plt.xlabel('Max signal')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel('SNR')
            plt.title('SNR vs max signal')
            plt.show()        

        if snr_threshold is None and snr_ratio is not None:
            snr_threshold = np.median(snr_spots)*snr_ratio
        print(f'SNR threshold: {snr_threshold}')

        if snr_threshold is not None:
            # Print the number of spots before filtering
            print(f'Number of spots before SNR filtering: {canidate_spots.shape[0]}')
            good_spots = [True if snr > snr_threshold else False for snr in snr_spots]
            canidate_spots = canidate_spots[good_spots, :]
            print(f'Number of spots after SNR filtering: {canidate_spots.shape[0]}')

        # decompose dense regions
        try: # TODO fix this try 
            spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                                                image=rna.astype(np.uint16), 
                                                spots=canidate_spots, 
                                                voxel_size=voxel_size_nm, 
                                                spot_radius=spot_size_nm if not use_log_hook else spot_radius_px,
                                                alpha=alpha,
                                                beta=beta,
                                                gamma=gamma)
        except RuntimeError:
            spots_post_decomposition = canidate_spots

        # TODO: define ts by some other metric for ts
        #         
        spots_post_clustering, clusters = detection.detect_clusters(
                                                        spots=spots_post_decomposition, 
                                                        voxel_size=voxel_size_nm, 
                                                        radius=CLUSTER_RADIUS, 
                                                        nb_min_spots=MIN_NUM_SPOT_FOR_CLUSTER)
        
        if sub_pixel_fitting:
            spots_subpx = detection.fit_subpixel(
                                        image=rna, 
                                        spots=spots_post_clustering, 
                                        voxel_size=voxel_size_nm, 
                                        spot_radius=voxel_size_nm)
        else:
            spots_subpx = None
            
        if verbose:
            print("detected canidate spots")
            print("\r shape: {0}".format(canidate_spots.shape))
            print("\r threshold: {0}".format(individual_thershold))
            print("detected spots after decomposition")
            print("\r shape: {0}".format(spots_post_decomposition.shape))
            print("detected spots after clustering")
            print("\r shape: {0}".format(spots_post_clustering.shape))
            print("detected clusters")
            print("\r shape: {0}".format(clusters.shape))

        if display_plots:
            plot.plot_elbow(
                images=rna, 
                voxel_size=voxel_size_nm if not use_log_hook else None, 
                spot_radius=spot_size_nm if not use_log_hook else None,
                log_kernel_size=spot_radius_px if use_log_hook else None,
                minimum_distance=spot_radius_px if use_log_hook else None)
            plot.plot_reference_spot(reference_spot, rescale=True)            
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                    canidate_spots, contrast=True)
                        
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                    spots_post_decomposition, contrast=True)
                        
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0), 
                                    spots=[spots_post_decomposition, clusters[:, :2] if len(rna.shape) == 2 else clusters[:, :3]], 
                                    shape=["circle", "circle"], 
                                    radius=[3, 6], 
                                    color=["red", "blue"],
                                    linewidth=[1, 2], 
                                    fill=[False, True], 
                                    contrast=True)
            
        return spots_post_clustering, dense_regions, reference_spot, clusters, spots_subpx, individual_thershold

    def standardize_df(self, df_cellresults, spots_px, spots_subpx, sub_pixel_fitting, clusters, c, timepoint, fov, independent_params, **kwargs):
            # merge spots_px and spots_um
            if spots_px.shape[1] == 6:
                if sub_pixel_fitting:
                    spots = np.concatenate([spots_px, spots_subpx], axis=1)
                    df_spotresults = pd.DataFrame(spots, columns=['z_px', 'y_px', 'x_px', 'cluster_index', 'z_nm', 'y_nm', 'x_nm', 'snr', 'signal'])
                    df_clusterresults = pd.DataFrame(clusters, columns=['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])

                else:
                    df_spotresults = pd.DataFrame(spots_px, columns=['z_px', 'y_px', 'x_px', 'cluster_index', 'snr', 'signal'])
                    df_clusterresults = pd.DataFrame(clusters, columns=['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])
            
            else:
                if sub_pixel_fitting:
                    spots = np.concatenate([spots_px, spots_subpx], axis=1)
                    df_spotresults = pd.DataFrame(spots, columns=['y_px', 'x_px', 'cluster_index', 'z_nm', 'y_nm', 'x_nm', 'snr', 'signal'])
                    df_clusterresults = pd.DataFrame(clusters, columns=['y_px', 'x_px', 'nb_spots', 'cluster_index'])

                else:
                    df_spotresults = pd.DataFrame(spots_px, columns=['y_px', 'x_px', 'cluster_index', 'snr', 'signal'])
                    df_clusterresults = pd.DataFrame(clusters, columns=['y_px', 'x_px', 'nb_spots', 'cluster_index'])

            df_spotresults['timepoint'] = [timepoint]*len(df_spotresults)
            df_spotresults['fov'] = [fov]*len(df_spotresults)
            df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

            if df_cellresults is not None:
                df_spotresults['timepoint'] = [timepoint]*len(df_spotresults)
                df_spotresults['fov'] = [fov]*len(df_spotresults)
                df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

            df_clusterresults['timepoint'] = [timepoint]*len(df_clusterresults)
            df_clusterresults['fov'] = [fov]*len(df_clusterresults)
            df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)

            return df_spotresults, df_clusterresults


class UFISH_SpotDetection_Step(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, image, FISHChannel, display_plots:bool = False, **kwargs):
        rna = image[:, :, :, FISHChannel[0]]
        rna = rna.squeeze()

        ufish = UFish()
        ufish.load_weights()

        pred_spots, enh_img = ufish.predict(rna)

        print(pred_spots)

        if display_plots:
            ufish.plot_result(np.max(rna, axis=0) if len(rna.shape) == 3 else rna, pred_spots)
            plt.show()
            plt.imshow(np.max(rna, axis=0) if len(rna.shape) == 3 else rna)
            plt.show()
            plt.imshow(np.max(enh_img, axis=0) if len(enh_img.shape) == 3 else enh_img)
            plt.show()


class TrackPy_SpotDetection(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, id, list_images, FISHChannel, nucChannel, spot_yx_px, spot_z_px, voxel_size_yx, voxel_size_z,
             map_id_imgprops, image_name:str, trackpy_minmass: float = None,  list_nuc_masks: list[np.array] = None, list_cell_masks: list[np.array] = None,
             trackpy_minsignal: float = None, trackpy_seperation_yx_px: float = 13, trackpy_seperation_z_px: float = 3, CLUSTER_RADIUS: float = 500, 
             MIN_NUM_SPOT_FOR_CLUSTER: int = 4,
             trackpy_maxsize: float = None, display_plots: bool = False, plot_types: list[str] = ['mass', 'size', 'signal', 'raw_mass'], 
             trackpy_percentile:int = 64, trackpy_use_pca: bool = False, verbose: bool = False, **kwargs):
        # Load in images and masks
        img = list_images[id]
        nuc = img[:, :, : , nucChannel[0]]
        nuc_label = list_nuc_masks[id] if list_nuc_masks is not None else None
        cell_label = list_cell_masks[id] if list_cell_masks is not None else None

        for c in range(len(FISHChannel)):
            fish = np.squeeze(img[:, :, :, FISHChannel[c]])   

            # 3D spot detection
            if len(fish.shape) == 3:
                spot_diameter = (spot_z_px, spot_yx_px, spot_yx_px)
                separation = (trackpy_seperation_z_px, trackpy_seperation_yx_px, trackpy_seperation_yx_px) if trackpy_seperation_yx_px is not None else None
                trackpy_features = tp.locate(fish, diameter=spot_diameter, minmass=trackpy_minmass, separation=separation)

            # 2D spot detection
            else:
                spot_diameter = spot_yx_px
                separation = trackpy_seperation_yx_px
                trackpy_features = tp.locate(fish, diameter=spot_diameter, separation=separation, percentile=trackpy_percentile)

            if trackpy_minmass is not None:
                trackpy_features = trackpy_features[trackpy_features['mass'] > trackpy_minmass]
            if trackpy_minsignal is not None:
                trackpy_features = trackpy_features[trackpy_features['signal'] > trackpy_minsignal]
            if trackpy_maxsize is not None:
                trackpy_features = trackpy_features[trackpy_features['size'] < trackpy_maxsize]

            # get np.array of spots for bigfish
            spots_px = np.array(
                trackpy_features[['z', 'y', 'x']].values if len(fish.shape) == 3 else trackpy_features[['y','x']].values
                )
            
            spots_nm = spots_px.copy()
            if len(fish.shape) == 3:
                spots_nm[:, 0] = spots_nm[:, 0]*voxel_size_z
                spots_nm[:, 1] = spots_nm[:, 1]*voxel_size_yx
                spots_nm[:, 2] = spots_nm[:, 2]*voxel_size_yx
            else:
                spots_nm[:, 0] = spots_nm[:, 0]*voxel_size_yx
                spots_nm[:, 1] = spots_nm[:, 1]*voxel_size_yx

            # Get cluster 
            spots_px, clusters = detection.detect_clusters(spots_px, 
                                                voxel_size=(voxel_size_z, voxel_size_yx, voxel_size_yx) if len(fish.shape) == 3 else (voxel_size_yx, voxel_size_yx), 
                                                radius=CLUSTER_RADIUS, 
                                                nb_min_spots=MIN_NUM_SPOT_FOR_CLUSTER)
            print(clusters)
            clusters = np.array(clusters)

            # Extract Cell level results
            if nuc_label is not None or cell_label is not None:
                cellresults = self.extract_cell_level_results(spots_px, clusters, nuc_label, cell_label, fish, nuc, 
                                                    verbose, display_plots)
                cellresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(cellresults)
                cellresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(cellresults)
                cellresults['FISH_Channel'] = [c]*len(cellresults)

            if trackpy_use_pca:
                pca_data = trackpy_features[['mass', 'size', 'signal', 'raw_mass']]
                scaler = StandardScaler()
                scaler.fit(pca_data)
                pca_data = scaler.transform(pca_data)

                pca = PCA(n_components=3)
                pca.fit(pca_data)
                X = pca.transform(pca_data)

                # color the spots best on the clusters
                kmeans_pca = KMeans(n_clusters=2)
                kmeans_pca.fit(X)
                trackpy_features['cluster'] = kmeans_pca.labels_
                plt.scatter(X[:, 0], X[:, 1], c=kmeans_pca.labels_)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.show()

                # print the PCA vectors and the explained variance
                print(['mass', 'size', 'signal', 'raw_mass'])
                print(pca.components_)
                print(pca.explained_variance_ratio_)

            # Plotting
            if display_plots:
                if len(fish.shape) == 3:
                    tp.annotate3d(trackpy_features, fish, plot_style={'markersize': 2})
                    tp.subpx_bias(trackpy_features)

                else:
                    plt.imshow(fish)
                    plt.show()
                    tp.annotate(trackpy_features, fish, plot_style={'markersize': 2})
                    tp.subpx_bias(trackpy_features)

                plot.plot_detection(fish if len(fish.shape) == 2 else np.max(fish, axis=0), 
                                    spots=[spots_px, clusters[:, :2] if len(fish.shape) == 2 else clusters[:, :3]], 
                                    shape=["circle", "circle"], 
                                    radius=[3, 6], 
                                    color=["red", "blue"],
                                    linewidth=[1, 2], 
                                    fill=[False, True], 
                                    contrast=True,
                                    path_output=os.path.join(self.step_output_dir, f'cluster_{image_name}') if self.step_output_dir is not None else None)
                
                for plot_type in plot_types:
                    fig, ax = plt.subplots()
                    ax.hist(trackpy_features[plot_type], bins=20)
                    # Optionally, label the axes.
                    ax.set(xlabel=plot_type, ylabel='count')
                    plt.show()

            # clean up output
            

            # rename x to x_px and y to y_px and z if it exists
            trackpy_features = trackpy_features.rename(columns={'x': 'x_px', 'y': 'y_px'})
            if len(fish.shape) == 3:
                trackpy_features = trackpy_features.rename(columns={'z': 'z_px'})

            # append frame number and fov number to the features
            trackpy_features['cluster_index'] = spots_px[:, -1]
            trackpy_features['frame'] = [map_id_imgprops[id]['tp_num']]*len(trackpy_features)
            trackpy_features['fov'] = [map_id_imgprops[id]['fov_num']]*len(trackpy_features)
            trackpy_features['FISH_Channel'] = [FISHChannel[0]]*len(trackpy_features)
            trackpy_features['x_nm'] = trackpy_features['x_px']*voxel_size_yx # convert to nm
            trackpy_features['y_nm'] = trackpy_features['y_px']*voxel_size_yx
            if len(fish.shape) == 3:
                trackpy_features['z_nm'] = trackpy_features['z_px']*voxel_size_z

        output = Trackpy_SpotDetection_Output(id=id, trackpy_features=trackpy_features)
        return output

#%% Axilary Steps
class Calculate_BIGFISH_Threshold(IndependentStepClass):
    def __init__(self):
        super().__init__()

    def main(self, images, FISHChannel: list[int], voxel_size_yx, voxel_size_z, spot_yx, spot_z, 
            MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD:int = 50,
            use_log_hook:bool =False, verbose:bool = False, 
            display_plots: bool = False, **kwargs):
        voxel_size = (float(voxel_size_z), float(voxel_size_yx), float(voxel_size_yx))
        spot_size = (float(spot_z), float(spot_yx), float(spot_yx))
        self.verbose = verbose
        self.display_plots = display_plots

        thresholds = []


        for c in FISHChannel:
            rna = images[:min(MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD, images.shape[0]), 0, c, :, :, :].compute()
            rna = [rna[r].astype(np.float32) for r in range(rna.shape[0])]
            spots, t = detection.detect_spots(images=rna, 
                                                        return_threshold=True, 
                                                        voxel_size=voxel_size,
                                                        spot_radius=spot_size if not use_log_hook else None)

            thresholds.append(t)
            
            print("Channel: ", c)
            print("Threshold: ", t)
            print()

        New_Parameters({'bigfish_threshold': thresholds})




#%% Masking
class DetectedSpot_Mask(SequentialStepsClass):
    def __init__(self):
        super().__init__()
    
    def main(self, masks, FISHChannel,
             timepoint, fov, df_spotresults: pd.DataFrame):
        for i, row in df_spotresults.iterrows():
            x, y = row['x_px'], row['y_px']
            try:
                masks[fov, timepoint, FISHChannel[0], 0, y-1:y+2, x-1:x+2] = 1
            except:
                pass
        
        New_Parameters({'masks': masks})
        

        






    



























