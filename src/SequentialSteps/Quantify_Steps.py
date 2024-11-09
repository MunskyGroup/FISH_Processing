import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from typing import Union
import pathlib
from bigfish import stack, detection, multistack, plot
import trackpy as tp
import tifffile

from SequentialSteps import SpotDetection_module  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import SequentialStepsClass, StepOutputsClass, SingleStepCompiler
from src.Util import Plots, SpotDetection

#%% Useful Functions
def add_indepenedent_params_to_df(df, independent_params):
    if df is None:
        return None
    if independent_params is None:
        return df
    else:
        for key, value in independent_params.items():
            df[key] = [value]*len(df)
        return df

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

#%%
class BIGFISH_SpotDetection(SequentialStepsClass):
    def __init__(self, step_name, step_output_dir, verbose=False):
        super().__init__(step_name, step_output_dir, verbose)
        self.snr_threshold = None
        self.snr_ratio = None

    def main(self, id, list_images, FISHChannel, nucChannel,
                voxel_size_yx, voxel_size_z, spot_yx, spot_z, map_id_imgprops, 
                image_name: str = None, list_nuc_masks: list[np.array] = None, list_cell_masks: list[np.array] = None,
                bigfish_threshold: Union[int, str] = None,
                bigfish_alpha: float = 0.7, bigfish_beta:float = 1, bigfish_gamma:float = 5, 
                CLUSTER_RADIUS:int = 500, MIN_NUM_SPOT_FOR_CLUSTER:int = 4, 
                verbose:bool = False, display_plots: bool = False,
                sub_pixel_fitting: bool = False, bigfish_minDistance:Union[float, list] = None, detect_elbow:bool = False,
                  **kwargs):

            # Load in images and masks
            nuc_label = list_nuc_masks[id] if list_nuc_masks is not None else None
            cell_label = list_cell_masks[id] if list_cell_masks is not None else None
            img = list_images[id]
            self.image_name = image_name
            nuc = img[:, :, :, nucChannel[0]]

            # cycle through FISH channels
            for c in range(len(FISHChannel)):
                rna = img[:, :, :, FISHChannel[c]]

                if detect_elbow:
                    threshold, thresholds, snr_results, spot_counts = self.dynamic_thresholding_with_snr(
                        rna, voxel_size_yx, voxel_size_z, spot_yx, spot_z, bigfish_threshold, 
                        bigfish_alpha, bigfish_beta, bigfish_gamma, CLUSTER_RADIUS, 
                        MIN_NUM_SPOT_FOR_CLUSTER, detect_elbow=detect_elbow, 
                        display_plots=display_plots
                    )
                else:
                    threshold = self._establish_threshold(c, bigfish_threshold, kwargs)

                spots_px, dense_regions, reference_spot, clusters, spots_subpx = self.bigfish_spotdetection(
                    rna=rna, voxel_size_yx=voxel_size_yx, voxel_size_z=voxel_size_z, spot_yx=spot_yx, spot_z=spot_z, alpha=bigfish_alpha,
                    beta=bigfish_beta, gamma=bigfish_gamma, CLUSTER_RADIUS=CLUSTER_RADIUS, MIN_NUM_SPOT_FOR_CLUSTER=MIN_NUM_SPOT_FOR_CLUSTER, 
                    threshold=threshold, verbose=verbose, display_plots=display_plots, sub_pixel_fitting=sub_pixel_fitting,
                    minimum_distance=bigfish_minDistance)
                
                # extract cell level results
                if nuc_label is not None or cell_label is not None:
                    df_cellresults = self.extract_cell_level_results(spots_px.astype(np.float64), clusters, nuc_label, cell_label, rna, nuc, 
                                                    verbose, display_plots)
                    df_cellresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_cellresults)
                    df_cellresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_cellresults)
                    df_cellresults['FISH_Channel'] = [c]*len(df_cellresults)

                else:
                    df_cellresults = None

                df_spotresults, df_clusterresults = self.standardize_df(spots_px, spots_subpx, sub_pixel_fitting, clusters, id, c, map_id_imgprops)

                df_spotresults = add_indepenedent_params_to_df(df_spotresults, kwargs['independent_params'])
                df_clusterresults = add_indepenedent_params_to_df(df_clusterresults, kwargs['independent_params'])
                df_cellresults = add_indepenedent_params_to_df(df_cellresults, kwargs['independent_params'])

                if c == 0:
                    df_spotresults_all = df_spotresults
                    df_clusterresults_all = df_clusterresults
                    df_cellresults_all = df_cellresults
                else:
                    df_spotresults_all = pd.concat([df_spotresults_all, df_spotresults])
                    df_clusterresults_all = pd.concat([df_clusterresults_all, df_clusterresults])
                    df_cellresults_all = pd.concat([df_cellresults_all, df_cellresults])

            output = SpotDetectionOutputClass(img_id=id, df_cellresults=df_cellresults_all, df_spotresults=df_spotresults_all, df_clusterresults=df_clusterresults_all)
            return output
        

    def bigfish_spotdetection(self, rna:np.array, voxel_size_yx:float, voxel_size_z:float, spot_yx:float, spot_z:float, alpha:int, beta:int,
                               gamma:int, CLUSTER_RADIUS:float, MIN_NUM_SPOT_FOR_CLUSTER:int, threshold:float, 
                               verbose: bool = False, display_plots: bool = False, sub_pixel_fitting: bool = False, minimum_distance:Union[list, float] = None, **kwargs):
        rna = rna.squeeze()

        voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
        spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))

        # Compute the spot radius in pixel
        spot_radius_px = detection.get_object_radius_pixel(
                voxel_size_nm=voxel_size_nm, 
                object_radius_nm=spot_size_nm, 
                ndim=3 if len(rna.shape) == 3 else 2)
        
        # LoG filter
        rna_log = stack.log_filter(rna,sigma=spot_radius_px)
       
        # local maximum detection
        mask = detection.local_maximum_detection(rna_log, min_distance=spot_radius_px)

        # Thresholding
        if threshold is None:
            # thresholding
            threshold = detection.automated_threshold_setting(rna_log, mask)
            canidate_spots, _ = detection.spots_thresholding(rna_log, mask, threshold)
            print("detected spots")
        
        # Compute SNR of the detected spots (2D array of coordinates)
        snr_spots, max_signal = compute_snr_spots(
            image=rna.astype(np.float64), 
            spots=canidate_spots.astype(np.float64), 
            voxel_size=voxel_size_nm, 
            spot_radius=spot_size_nm)
        
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

        if self.snr_threshold is not None:
            good_spots = [True if snr > self.snr_threshold else False for snr in snr_spots]
            canidate_spots = canidate_spots[good_spots, :]
            print(f'Number of spots after SNR filtering: {canidate_spots.shape[0]}')

        spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                                                image=rna.astype(np.uint16), 
                                                spots=canidate_spots, 
                                                voxel_size=voxel_size_nm, 
                                                spot_radius=spot_size_nm,
                                                alpha=alpha,
                                                beta=beta,
                                                gamma=gamma)

        
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
                voxel_size=voxel_size_nm, 
                spot_radius=spot_size_nm ,
                log_kernel_size=spot_radius_px,
                minimum_distance=spot_radius_px,
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

        # # Extract cells and associated spots
        # image_contrasted = np.max(rna, axis=0) if len(rna.shape) == 3 else rna
        # image_contrasted = stack.rescale(image_contrasted,channel_to_stretch=None)
        # print(image_contrasted.shape)


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
                    image=image_contrasted, cell_mask=cell_mask, nuc_mask=nuc_mask, rescale=True, contrast=True,
                    title="Cell {0}".format(i), 
                    path_output=os.path.join(self.step_output_dir, f'cell_{self.image_name}_cell{i}') if self.step_output_dir is not None else None)

        df = multistack.summarize_extraction_results(fov_results, ndim=3)
        return df

    def get_spot_properties(self, rna, spot, voxel_size_yx, voxel_size_z, spot_yx, spot_z):
        pass

    def get_cluster_properties(self, rna, cluster, voxel_size_yx, voxel_size_z, spot_yx, spot_z):
        pass

    def standardize_df(self, spots_px, spots_subpx, sub_pixel_fitting, clusters, id, c, map_id_imgprops):
            # merge spots_px and spots_um
            if spots_px.shape[1] == 4:
                if sub_pixel_fitting:
                    spots = np.concatenate([spots_px, spots_subpx], axis=1)
                    df_spotresults = pd.DataFrame(spots, columns=['z_px', 'y_px', 'x_px', 'cluster_index', 'z_nm', 'y_nm', 'x_nm'])
                    df_clusterresults = pd.DataFrame(clusters, columns=['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])

                else:
                    df_spotresults = pd.DataFrame(spots_px, columns=['z_px', 'y_px', 'x_px', 'cluster_index'])
                    df_clusterresults = pd.DataFrame(clusters, columns=['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])
            
            else:
                if sub_pixel_fitting:
                    spots = np.concatenate([spots_px, spots_subpx], axis=1)
                    df_spotresults = pd.DataFrame(spots, columns=['y_px', 'x_px', 'cluster_index', 'z_nm', 'y_nm', 'x_nm'])
                    df_clusterresults = pd.DataFrame(clusters, columns=['y_px', 'x_px', 'nb_spots', 'cluster_index'])

                else:
                    df_spotresults = pd.DataFrame(spots_px, columns=['y_px', 'x_px', 'cluster_index'])
                    df_clusterresults = pd.DataFrame(clusters, columns=['y_px', 'x_px', 'nb_spots', 'cluster_index'])

            df_spotresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_spotresults)
            df_spotresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_spotresults)
            df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)
            df_spotresults['img_id'] = [id]*len(df_spotresults)

            df_clusterresults['timepoint'] = [map_id_imgprops[id]['tp_num']]*len(df_clusterresults)
            df_clusterresults['fov'] = [map_id_imgprops[id]['fov_num']]*len(df_clusterresults)
            df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)
            df_clusterresults['img_id'] = [id]*len(df_clusterresults)


            return df_spotresults, df_clusterresults