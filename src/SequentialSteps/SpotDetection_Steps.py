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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import SequentialStepsClass, StepOutputsClass, SingleStepCompiler
from src.Util import Plots, SpotDetection


#%% Output Classes
class SpotDetectionOutputClass(StepOutputsClass):
    def __init__(self, img_id, df_cellresults, df_spotresults, df_clusterresults):
        self.img_id = [img_id]
        self.df_cellresults = df_cellresults
        self.df_spotresults = df_spotresults
        self.df_clusterresults = df_clusterresults

    def append(self, newOutput):
        self.img_id = [*self.img_id, *newOutput.img_id]
    
        if self.df_cellresults is None:
            self.df_cellresults = newOutput.df_cellresults
        else:
            self.df_cellresults = pd.concat([self.df_cellresults, newOutput.df_cellresults])

        self.df_spotresults = pd.concat([self.df_spotresults, newOutput.df_spotresults])
        self.df_clusterresults = pd.concat([self.df_clusterresults, newOutput.df_clusterresults])


class SpotDetectionStepOutputClass(StepOutputsClass):
    def __init__(self, id, individual_threshold_spot_detection, avg_number_of_spots_per_cell_each_ch, dfFISH):
        self.individual_threshold_spot_detection = [individual_threshold_spot_detection]
        self.img_id = [id]
        self.avg_number_of_spots_per_cell_each_ch = [avg_number_of_spots_per_cell_each_ch]
        self.dfFISH = dfFISH

    def append(self, newOutputs):
        self.individual_threshold_spot_detection = self.individual_threshold_spot_detection + newOutputs.individual_threshold_spot_detection
        self.img_id = self.img_id + newOutputs.img_id
        self.avg_number_of_spots_per_cell_each_ch = self.avg_number_of_spots_per_cell_each_ch + newOutputs.avg_number_of_spots_per_cell_each_ch
        self.dfFISH = newOutputs.dfFISH  # I believe it does this in place :(


class Trackpy_SpotDetection_Output(StepOutputsClass):
    def __init__(self, id, trackpy_features):
        super().__init__()
        self.id = [id]
        self.trackpy_features = trackpy_features

    def append(self, newOutput):
        self.id = [*self.id, *newOutput.id]
        self.trackpy_features = pd.concat([self.trackpy_features, newOutput.trackpy_features])


#%% Step Classes
class BIGFISH_SpotDetection(SequentialStepsClass):
    """
    A class for detecting RNA spots in FISH images using the BIGFISH library.
    Methods
    -------
    __init__():
        Initializes the BIGFISH_SpotDetection class.
    main(id, list_images, FISHChannel, nucChannel, voxel_size_yx, voxel_size_z, spot_yx, spot_z, map_id_imgprops, image_name=None, list_nuc_masks=None, list_cell_masks=None, bigfish_mean_threshold=None, bigfish_alpha=0.7, bigfish_beta=1, bigfish_gamma=5, CLUSTER_RADIUS=500, MIN_NUM_SPOT_FOR_CLUSTER=4, use_log_hook=False, verbose=False, display_plots=False, **kwargs):
        Main method to detect spots in FISH images and extract cell-level results.
        Parameters:
        - id (int): Identifier for the image.
        - list_images (list): List of images.
        - FISHChannel (list): List of FISH channels.
        - nucChannel (list): List of nuclear channels.
        - voxel_size_yx (float): Voxel size in the yx plane.
        - voxel_size_z (float): Voxel size in the z plane.
        - spot_yx (float): Spot size in the yx plane.
        - spot_z (float): Spot size in the z plane.
        - map_id_imgprops (dict): Mapping of image properties.
        - image_name (str, optional): Name of the image.
        - list_nuc_masks (list[np.array], optional): List of nuclear masks.
        - list_cell_masks (list[np.array], optional): List of complete cell masks.
        - bigfish_mean_threshold (list[float], optional): List of mean thresholds for spot detection.
        - bigfish_alpha (float, optional): Alpha parameter for spot decomposition.
        - bigfish_beta (float, optional): Beta parameter for spot decomposition.
        - bigfish_gamma (float, optional): Gamma parameter for spot decomposition.
        - CLUSTER_RADIUS (float, optional): Radius for clustering spots.
        - MIN_NUM_SPOT_FOR_CLUSTER (int, optional): Minimum number of spots for clustering.
        - use_log_hook (bool, optional): Whether to use log kernel for spot detection.
        - verbose (bool, optional): Whether to print verbose output.
        - display_plots (bool, optional): Whether to display plots.
    """
    def __init__(self):
        super().__init__()

    def main(self, id, list_images, FISHChannel,  nucChannel,
             voxel_size_yx, voxel_size_z, spot_yx, spot_z, map_id_imgprops, 
             image_name: str = None, list_nuc_masks: list[np.array] = None, list_cell_masks: list[np.array] = None,
             bigfish_threshold: Union[int, str] = None, bigfish_alpha: float = 0.7, bigfish_beta:float = 1, bigfish_gamma:float = 5, 
             CLUSTER_RADIUS:int = 500, MIN_NUM_SPOT_FOR_CLUSTER:int = 4, use_log_hook:bool = False, 
             verbose:bool = False, display_plots: bool = False, bigfish_use_pca: bool = False,
             sub_pixel_fitting: bool = False, bigfish_minDistance:Union[float, list] = None, **kwargs):

        # Load in images and masks
        nuc_label = list_nuc_masks[id] if list_nuc_masks is not None else None
        cell_label = list_cell_masks[id] if list_cell_masks is not None else None
        img = list_images[id]
        self.image_name = image_name

        # cycle through FISH channels
        for c in range(len(FISHChannel)):
            # extract single rna channel
            rna = img[:, :, :, FISHChannel[c]]
            nuc = img[:, :, :, nucChannel[0]]


            threshold = self._establish_threshold(c, bigfish_threshold, kwargs)

            # detect spots
            spots_px, dense_regions, reference_spot, clusters, spots_subpx = self.bigfish_spotdetection(
                rna=rna, voxel_size_yx=voxel_size_yx, voxel_size_z=voxel_size_z, spot_yx=spot_yx, spot_z=spot_z, alpha=bigfish_alpha,
                beta=bigfish_beta, gamma=bigfish_gamma, CLUSTER_RADIUS=CLUSTER_RADIUS, MIN_NUM_SPOT_FOR_CLUSTER=MIN_NUM_SPOT_FOR_CLUSTER, 
                threshold=threshold, use_log_hook=use_log_hook, verbose=verbose, display_plots=display_plots, sub_pixel_fitting=sub_pixel_fitting,
                minimum_distance=bigfish_minDistance, use_pca=bigfish_use_pca)

            # extract cell level results
            if nuc_label is not None or cell_label is not None:
                df = self.extract_cell_level_results(spots_px, clusters, nuc_label, cell_label, rna, nuc, 
                                                 verbose, display_plots)
                df['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df)
                df['fov'] = [map_id_imgprops[id]['fov_num']]*len(df)
                df['FISH_Channel'] = [c]*len(df)

            else:
                df = None

            # merge spots_px and spots_um
            if spots_px.shape[1] == 4:
                if sub_pixel_fitting:
                    spots = np.concatenate([spots_px, spots_subpx], axis=1)
                    
                    df_spotresults = pd.DataFrame(spots, columns=['z_px', 'y_px', 'x_px', 'cluster_index', 'z_nm', 'y_nm', 'x_nm'])
                    df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
                    df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
                    df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)
                    df_spotresults['img_id'] = [id]*len(df_spotresults)

                    df_clusterresults = pd.DataFrame(clusters, columns=['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])
                    df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
                    df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
                    df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)
                    df_clusterresults['img_id'] = [id]*len(df_clusterresults)

                else:
                    df_spotresults = pd.DataFrame(spots_px, columns=['z_px', 'y_px', 'x_px', 'cluster_index'])
                    df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
                    df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
                    df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)
                    df_spotresults['img_id'] = [id]*len(df_spotresults)

                    df_clusterresults = pd.DataFrame(clusters, columns=['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])
                    df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
                    df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
                    df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)
                    df_clusterresults['img_id'] = [id]*len(df_clusterresults)
            
            else:
                if sub_pixel_fitting:
                    spots = np.concatenate([spots_px, spots_subpx], axis=1)
                    
                    df_spotresults = pd.DataFrame(spots, columns=['y_px', 'x_px', 'cluster_index', 'z_nm', 'y_nm', 'x_nm'])
                    df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
                    df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
                    df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)
                    df_spotresults['img_id'] = [id]*len(df_spotresults)

                    df_clusterresults = pd.DataFrame(clusters, columns=['y_px', 'x_px', 'nb_spots', 'cluster_index'])
                    df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
                    df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
                    df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)
                    df_clusterresults['img_id'] = [id]*len(df_clusterresults)

                else:
                    df_spotresults = pd.DataFrame(spots_px, columns=['y_px', 'x_px', 'cluster_index'])
                    df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
                    df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
                    df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)
                    df_spotresults['img_id'] = [id]*len(df_spotresults)

                    df_clusterresults = pd.DataFrame(clusters, columns=['y_px', 'x_px', 'nb_spots', 'cluster_index'])
                    df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
                    df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
                    df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)
                    df_clusterresults['img_id'] = [id]*len(df_clusterresults)

            # create output object
            output = SpotDetectionOutputClass(img_id=id, df_cellresults=df, df_spotresults=df_spotresults, df_clusterresults=df_clusterresults)
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

    def bigfish_spotdetection(self, rna:np.array, voxel_size_yx:float, voxel_size_z:float, spot_yx:float, spot_z:float, alpha:int, beta:int,
                               gamma:int, CLUSTER_RADIUS:float, MIN_NUM_SPOT_FOR_CLUSTER:int, threshold:float, use_log_hook:bool, 
                               verbose: bool = False, display_plots: bool = False, sub_pixel_fitting: bool = False, minimum_distance:Union[list, float] = None,
                               use_pca: bool = False):
        rna = rna.squeeze()

        voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
        spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))

        if use_log_hook:
            spot_radius_px = detection.get_object_radius_pixel(
                voxel_size_nm=voxel_size_nm, 
                object_radius_nm=spot_size_nm, 
                ndim=3 if len(rna.shape) == 3 else 2)
        else:
            spot_radius_px = None
            

        canidate_spots = detection.detect_spots(
                                            images=rna, 
                                            return_threshold=False, 
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

        # decompose dense regions
        spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                                                image=rna.astype(np.uint16), 
                                                spots=canidate_spots, 
                                                voxel_size=voxel_size_nm, 
                                                spot_radius=spot_size_nm if not use_log_hook else spot_radius_px,
                                                alpha=alpha,
                                                beta=beta,
                                                gamma=gamma)

        # TODO: define ts by some other metric for ts


        

        
        spots_post_clustering, clusters = detection.detect_clusters(
                                                        spots=spots_post_decomposition, 
                                                        voxel_size=voxel_size_nm, 
                                                        radius=CLUSTER_RADIUS, 
                                                        nb_min_spots=MIN_NUM_SPOT_FOR_CLUSTER)
        
        if sub_pixel_fitting:
            spots_subpx = detection.fit_subpixel(
                                        image=rna, 
                                        spots=canidate_spots, 
                                        voxel_size=voxel_size_nm, 
                                        spot_radius=voxel_size_nm)
        else:
            spots_subpx = None
            
        if verbose:
            print("detected canidate spots")
            print("\r shape: {0}".format(canidate_spots.shape))
            print("\r threshold: {0}".format(threshold))
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
                minimum_distance=spot_radius_px if use_log_hook else None,
                path_output=os.path.join(self.step_output_dir, f'elbow_{self.image_name}') if self.step_output_dir is not None else None)
            plot.plot_reference_spot(reference_spot, rescale=True, 
                                    path_output=os.path.join(self.step_output_dir, f'reference_spot_{self.image_name}') if self.step_output_dir is not None else None)
            
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                    canidate_spots, contrast=True, 
                                    path_output=os.path.join(self.step_output_dir, f'canidate_{self.image_name}') if self.step_output_dir is not None else None)
            
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                    spots_post_decomposition, contrast=True, 
                                    path_output=os.path.join(self.step_output_dir, f'detection_{self.image_name}') if self.step_output_dir is not None else None)
            
            plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0), 
                                    spots=[spots_post_decomposition, clusters[:, :2] if len(rna.shape) == 2 else clusters[:, :3]], 
                                    shape=["circle", "circle"], 
                                    radius=[3, 6], 
                                    color=["red", "blue"],
                                    linewidth=[1, 2], 
                                    fill=[False, True], 
                                    contrast=True,
                                    path_output=os.path.join(self.step_output_dir, f'cluster_{self.image_name}') if self.step_output_dir is not None else None)
        return spots_post_clustering, dense_regions, reference_spot, clusters, spots_subpx

    def extract_cell_level_results(self, spots, clusters, nuc_label, cell_label, rna, nuc, verbose, display_plots):
        # convert masks to max projection
        if nuc_label is not None and len(nuc_label.shape) != 2:
            nuc_label = np.max(nuc_label, axis=0)
        if cell_label is not None and len(cell_label.shape) != 2:
            cell_label = np.max(cell_label, axis=0)

        # remove transcription sites
        spots_no_ts, foci, ts = multistack.remove_transcription_site(spots, clusters, nuc_label, ndim=3)
        if verbose:
            print("detected spots (without transcription sites)")
            print("\r shape: {0}".format(spots_no_ts.shape))
            print("\r dtype: {0}".format(spots_no_ts.dtype))

        # get spots inside and outside nuclei
        spots_in, spots_out = multistack.identify_objects_in_region(nuc_label, spots, ndim=3)
        if verbose:
            print("detected spots (inside nuclei)")
            print("\r shape: {0}".format(spots_in.shape))
            print("\r dtype: {0}".format(spots_in.dtype), "\n")
            print("detected spots (outside nuclei)")
            print("\r shape: {0}".format(spots_out.shape))
            print("\r dtype: {0}".format(spots_out.dtype))

        # extract fov results
        other_images = {}
        other_images["dapi"] = np.max(nuc, axis=0).astype("uint16") if nuc is not None else None
        fov_results = multistack.extract_cell(
            cell_label=cell_label.astype("uint16") if cell_label is not None else nuc_label.astype("uint16"),
            ndim=3,
            nuc_label=nuc_label.astype("uint16"),
            rna_coord=spots_no_ts,
            others_coord={"foci": foci, "transcription_site": ts},
            image=np.max(rna, axis=0).astype("uint16"),
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
                    image=image_contrasted, cell_mask=cell_mask, nuc_mask=nuc_mask,
                    title="Cell {0}".format(i), 
                    path_output=os.path.join(self.step_output_dir, f'cell_{self.image_name}_cell{i}') if self.step_output_dir is not None else None)

        df = multistack.summarize_extraction_results(fov_results, ndim=3)
        return df

    def get_spot_properties(self, rna, spot, voxel_size_yx, voxel_size_z, spot_yx, spot_z):
        pass


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

    def main(self, id, list_images, FISHChannel, spot_yx_px, spot_z_px, voxel_size_yx, voxel_size_z,
             map_id_imgprops, trackpy_minmass: float = None,  trackpy_minsignal: float = None, 
             trackpy_seperation_yx_px: float = 13, trackpy_seperation_z_px: float = 3, trackpy_maxsize: float = None,
               display_plots: bool = False, plot_types: list[str] = ['mass', 'size', 'signal', 'raw_mass'], 
               trackpy_percentile:int = 64, trackpy_use_pca: bool = False, **kwargs):
        # Load in image and extract FISH channel
        img = list_images[id]
        fish = np.squeeze(img[:, :, :, FISHChannel[0]])   

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
            
            for plot_type in plot_types:
                fig, ax = plt.subplots()
                ax.hist(trackpy_features[plot_type], bins=20)
                # Optionally, label the axes.
                ax.set(xlabel=plot_type, ylabel='count')
                plt.show()

        # append frame number and fov number to the features
        trackpy_features['frame'] = [map_id_imgprops[id]['tp_num']]*len(trackpy_features)
        trackpy_features['fov'] = [map_id_imgprops[id]['fov_num']]*len(trackpy_features)
        trackpy_features['FISH_Channel'] = [FISHChannel[0]]*len(trackpy_features)
        trackpy_features['xum'] = trackpy_features['x']*voxel_size_yx/1000 # convert to microns
        trackpy_features['yum'] = trackpy_features['y']*voxel_size_yx/1000
        if len(fish.shape) == 3:
            trackpy_features['zum'] = trackpy_features['z']*voxel_size_z/1000

        output = Trackpy_SpotDetection_Output(id=id, trackpy_features=trackpy_features)
        return output





class SpotDetectionStepClass_Luis(SequentialStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self,
             id: int,
             list_images: list[np.array],
             cytoChannel: list[int],
             nucChannel: list[int],
             FISHChannel: list[int],
             list_image_names: list,
             temp_folder_name: Union[str, pathlib.Path],
             threshold_for_spot_detection: float,
             segmentation_successful: list[bool],
             CLUSTER_RADIUS: float,
             minimum_spots_cluster: float,
             list_cell_masks: list[np.array],
             list_nuc_masks: list[np.array],
             list_cyto_masks: list[np.array],
             voxel_size_z: float,
             voxel_size_yx: float,
             psf_z: float,
             psf_yx: float,
             save_all_images: bool,
             filtered_folder_name: Union[str, pathlib.Path],
             show_plots: bool = True,
             display_spots_on_multiple_z_planes: bool = False,
             use_log_filter_for_spot_detection: bool = False,
             save_files: bool = True,
             **kwargs) -> SpotDetectionStepOutputClass:

        img = list_images[id]
        list_cell_masks = list_cell_masks[id]
        list_nuc_masks = list_nuc_masks[id]
        list_cyto_masks = list_cyto_masks[id]
        file_name = list_image_names[id]
        temp_folder_name = temp_folder_name
        segmentation_successful = segmentation_successful[id]
        list_voxels = [voxel_size_z, voxel_size_yx]
        list_psfs = [psf_z, psf_yx]

        temp_file_name = file_name[:file_name.rfind(
            '.')]  # slicing the name of the file. Removing after finding '.' in the string.
        temp_original_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                    'ori_' + temp_file_name + '.png')

        temp_segmentation_img_name = pathlib.Path().absolute().joinpath(temp_folder_name,
                                                                        'seg_' + temp_file_name + '.png')
        # Modified Luis's Code
        if segmentation_successful:
            temp_detection_img_name = pathlib.Path().absolute().joinpath(temp_folder_name, 'det_' + temp_file_name)
            dataframe_FISH, list_fish_images, thresholds_spot_detection = (
                SpotDetection(img,
                              FISHChannel,
                              cytoChannel,
                              nucChannel,
                              cluster_radius=CLUSTER_RADIUS,
                              minimum_spots_cluster=minimum_spots_cluster,
                              list_cell_masks=list_cell_masks,
                              list_nuc_masks=list_nuc_masks,
                              list_cyto_masks_no_nuclei=list_cyto_masks,
                              dataframe=self.dataframe,
                              image_counter=id,
                              list_voxels=list_voxels,
                              list_psfs=list_psfs,
                              show_plots=show_plots,
                              image_name=temp_detection_img_name,
                              save_all_images=save_all_images,
                              display_spots_on_multiple_z_planes=display_spots_on_multiple_z_planes,
                              use_log_filter_for_spot_detection=use_log_filter_for_spot_detection,
                              threshold_for_spot_detection=threshold_for_spot_detection,
                              save_files=save_files,
                              **kwargs).get_dataframe())

            self.dataframe = dataframe_FISH
            # print(dataframe)

            print('    Intensity threshold for spot detection : ', str(thresholds_spot_detection))
            # Create the image with labels.
            df_test = self.dataframe.loc[self.dataframe['image_id'] == id]
            test_cells_ids = np.unique(df_test['cell_id'].values)
            # Saving the average number of spots per cell
            list_number_of_spots_per_cell_for_each_spot_type = []
            list_max_number_of_spots_per_cell_for_each_spot_type = []
            for sp in range(len(FISHChannel)):
                detected_spots = np.asarray([len(self.dataframe.loc[(self.dataframe['cell_id'] == cell_id_test) & (
                        self.dataframe['spot_type'] == sp) & (self.dataframe['is_cell_fragmented'] != -1)].spot_id)
                                             for i, cell_id_test in enumerate(test_cells_ids)])
                average_number_of_spots_per_cell = int(np.mean(detected_spots))
                max_number_of_spots_per_cell = int(np.max(detected_spots))
                list_number_of_spots_per_cell_for_each_spot_type.append(average_number_of_spots_per_cell)
                list_max_number_of_spots_per_cell_for_each_spot_type.append(max_number_of_spots_per_cell)
            print('    Average detected spots per cell :        ', list_number_of_spots_per_cell_for_each_spot_type)
            print('    Maximum detected spots per cell :        ', list_max_number_of_spots_per_cell_for_each_spot_type)
            #list_average_spots_per_cell.append(list_number_of_spots_per_cell_for_each_spot_type)
            # saving FISH images
            if save_all_images:
                for j in range(len(FISHChannel)):
                    filtered_image_path = pathlib.Path().absolute().joinpath(filtered_folder_name, 'filter_Ch_' + str(
                        FISHChannel[j]) + '_' + temp_file_name + '.tif')
                    tifffile.imwrite(filtered_image_path, list_fish_images[j])
            # Create the image with labels.
            df_subset = dataframe_FISH.loc[dataframe_FISH['image_id'] == id]
            df_labels = df_subset.drop_duplicates(subset=['cell_id'])
            # Plotting cells 
            if save_files:
                Plots().plotting_masks_and_original_image(image=img,
                                                          list_cell_masks=list_cell_masks,
                                                          list_nuc_masks=list_nuc_masks,
                                                          channels_with_cytosol=cytoChannel,
                                                          channels_with_nucleus=nucChannel,
                                                          image_name=temp_segmentation_img_name,
                                                          show_plots=show_plots,
                                                          df_labels=df_labels)
            # del list_cell_masks, list_nuc_masks, list_cyto_masks_no_nuclei, list_fish_images,df_subset,df_labels
        else:
            raise Exception('Segmentation was not successful, so spot detection was not performed.')

        # OUTPUTS:
        output = SpotDetectionStepOutputClass(id=id,
                                              individual_threshold_spot_detection=thresholds_spot_detection,
                                              avg_number_of_spots_per_cell_each_ch=list_number_of_spots_per_cell_for_each_spot_type,
                                              dfFISH=self.dataframe)
        return output

    def first_run(self, id):
        self.dataframe = None
if __name__ == "__main__":
    ds = Dataset(r"C:\Users\Jack\Desktop\H128_Tiles_100ms_5mW_Blue_15x15_10z_05step_2")
    kwargs = {'nucChannel': [0], 'FISHChannel': [0],
          'user_select_number_of_images_to_run': 5}
    compiler = SingleStepCompiler(ds, kwargs)
    output = compiler.sudo_run_step(UFISH_SpotDetection_Step)
    



























