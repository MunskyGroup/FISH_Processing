import os

import numpy as np

from bigfish import detection, plot

from .. import PipelineDataClass, PipelineSettings, Experiment, ScopeClass, prePipelineStepsClass, StepOutputsClass

from src.Util.Utilities import Utilities
from src.Util.BigFish import BigFISH

#%% Output Directories
class Make_Analysis_Dir_Output_JF(StepOutputsClass):
    def __init__(self, analysis_dir):
        super().__init__()
        self.analysis_location = analysis_dir

class Make_Analysis_Dir_JF(prePipelineStepsClass):
    def __init__(self):
        super().__init__()
        self.ModifyPipelineData = False

    def main(self, initial_data_location,
             **kwargs):

        cwd = os.getcwd()

        analysis_dir = os.path.join(cwd, f'analysis_{os.path.basename(initial_data_location)}')

        if not os.path.exists(analysis_dir):
            os.mkdir(analysis_dir)

        return Make_Analysis_Dir_Output_JF(analysis_dir)

class Make_Output_Dir_Output_JF(StepOutputsClass):
    def __init__(self, output_dir):
        super().__init__()
        self.output_location = output_dir

class Make_Output_Dir_JF(prePipelineStepsClass):
    def __init__(self):
        super().__init__()
        self.ModifyPipelineData = False

    def main(self,
             initial_data_location,
             **kwargs):

        cwd = os.getcwd()

        output_dir = os.path.join(cwd, f'output_{os.path.basename(initial_data_location)}')

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        return Make_Output_Dir_Output_JF(output_dir)


class Make_Output_Dir_Output(StepOutputsClass):
    def __init__(self, output_dir):
        super().__init__()
        self.output_identification_string = output_dir


class Make_Output_Dir(prePipelineStepsClass):
    def __init__(self):
        super().__init__()
        self.ModifyPipelineData = False

    def main(self,
             initial_data_location,
             diameter_nucleus,
             diameter_cytosol,
             spot_z,
             spot_yx,
             FISHChannel,
             threshold_for_spot_detection,
             save_files=True,
             **kwargs):

        # Creating folder to store outputs.
        if save_files:
            output_identification_string = Utilities().create_output_folders(initial_data_location,
                                                                             diameter_nucleus,
                                                                             diameter_cytosol, spot_z,
                                                                             spot_yx,
                                                                             threshold_for_spot_detection,
                                                                             FISHChannel,
                                                                             threshold_for_spot_detection)
        else:
            output_identification_string = ''

        return Make_Output_Dir_Output(output_identification_string)

#%% Consolidate Image Shapes
class ConsolidateImageShapesOutput(StepOutputsClass):
    def __init__(self, list_images) -> None:
        super().__init__()
        self.ModifyPipelineData = True
        self.list_images = list_images


class ConsolidateImageShapes(prePipelineStepsClass):
    """
    This consolidates all images in list_images to the same shape (Z, Y, X, C)

   Inputs:
   PipelineData:
   - list_images

   Returns:
   ConsolidateImageShapesOutput (stored in pipelineData):
   - list_images (modifies PipelineData)
    """

    def __init__(self) -> None:
        super().__init__()
        self.ModifyPipelineData = True

    def main(self,
             list_images,
             **kwargs):
        # Luis's code
        if len(list_images[0].shape) < 3:
            list_images_extended = [np.expand_dims(img, axis=0) for img in list_images]
            list_images = list_images_extended

        if len(list_images[0].shape) < 4:
            list_images_extended = [np.expand_dims(img, axis=0) for img in list_images]
            list_images = list_images_extended
        else:
            list_images = list_images

        # outputs
        output = ConsolidateImageShapesOutput(list_images=list_images)
        return output

#%% Calculate Sharpness
class CalculateSharpnessOutput(StepOutputsClass):
    def __init__(self, list_is_image_sharp, list_images, list_metric_sharpness_images) -> None:
        super().__init__()
        self.ModifyPipelineData = True
        self.list_is_image_sharp = list_is_image_sharp
        self.list_images = list_images
        self.list_metric_sharpness_images = list_metric_sharpness_images


class CalculateSharpness(prePipelineStepsClass):
    """
    This step remove entire images if they are deemed out of focus (not just single Zs)

   Inputs:
   PipelineSettings:
   - remove_out_of_focus_images: flag to remove out of focus images
   - sharpness_threshold: Threshold for removing out of focus images/ deeming out of focus images (1.10 normally works
   well)

   PipelineData:
   - list_images

   Experiment:
   - FISH Channel

   Returns:
   CalculateSharpnessOutput (stored in pipelineData):
   - list_is_image_sharp
   - list_metric_sharpness_images
   - list_images (modified)
    """

    def __init__(self) -> None:
        super().__init__()

    def main(self,
             list_images,
             FISHChannel,
             sharpness_threshold,
             remove_out_of_focus_images, **kwargs):
        # Modified Luis's Code
        # this can be rewritten to be more efficient, calculate sharpness is ran either way, we just change the outputs
        if remove_out_of_focus_images:
            list_metric_sharpness_images, list_is_image_sharp, list_sharp_images = Utilities().calculate_sharpness(
                list_images, channels_with_FISH=FISHChannel, threshold=sharpness_threshold)
        else:
            list_is_image_sharp = np.ones(len(list_images))
            list_is_image_sharp = [bool(x) for x in list_is_image_sharp]
            list_metric_sharpness_images = Utilities().calculate_sharpness(list_images,
                                                                           channels_with_FISH=FISHChannel,
                                                                           threshold=sharpness_threshold)[0]
            list_sharp_images = None

        # outputs
        output = CalculateSharpnessOutput(list_is_image_sharp=list_is_image_sharp, list_images=list_sharp_images,
                                          list_metric_sharpness_images=list_metric_sharpness_images)

        return output

#%% Automatic Spot Detection
class AutomaticThresholdingOutput_JF(StepOutputsClass):
    def __init__(self, bigfish_min_threshold, bigfish_max_threshold, bigfish_mean_threshold, bigfish_std_threshold) -> None:
        super().__init__()
        self.ModifyPipelineData = False
        self.bigfish_min_threshold = bigfish_min_threshold
        self.bigfish_max_threshold = bigfish_max_threshold
        self.bigfish_mean_threshold = bigfish_mean_threshold
        self.bigfish_std_threshold = bigfish_std_threshold


class AutomaticSpotDetection_JF(prePipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, list_images, FISHChannel: list[int], voxel_size_yx, voxel_size_z, spot_yx, spot_z, 
             list_image_names, MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD:int = 50,
              use_log_hook:bool =False, bigfish_min_threshold:float = 0, verbose:bool = False, **kwargs):
        voxel_size = (float(voxel_size_z), float(voxel_size_yx), float(voxel_size_yx))
        spot_size = (float(spot_z), float(spot_yx), float(spot_yx))
        output_dir = self.step_output_dir
        self.verbose = verbose

        min_thresholds = []
        max_thresholds = []
        mean_thresholds = []
        std_thresholds = []
        for c in FISHChannel:
            num_images_used = 0
            list_thresholds = []
            for id, img in enumerate(list_images):
                image_name = list_image_names[id]
                image_name = os.path.splitext(image_name)[0]
                rna = img[:,:,:,c]
                if output_dir is not None:
                    self.step_output_dir = os.path.join(output_dir, f'elbow_{image_name}')
                threshold = self.calculate_threshold(rna, voxel_size, spot_size, use_log_hook)

                if threshold is not None and threshold > bigfish_min_threshold:
                    if verbose:
                        print("Threshold: ", threshold)
                    list_thresholds.append(threshold)
                    num_images_used += 1
                else:
                    if verbose:
                        print("Threshold: ", threshold, " was regected")

                if num_images_used >= MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD:
                    break
            
            min_thresholds.append(np.min(list_thresholds))
            max_thresholds.append(np.max(list_thresholds))
            mean_thresholds.append(np.mean(list_thresholds))
            std_thresholds.append(np.std(list_thresholds))
            if verbose:
                print("Channel: ", c)
                print("Min Threshold: ", min_thresholds[-1])
                print("Max Threshold: ", max_thresholds[-1])
                print("Mean Threshold: ", mean_thresholds[-1])
                print("Std Threshold: ", std_thresholds[-1])

        output = AutomaticThresholdingOutput_JF(bigfish_min_threshold=min_thresholds, bigfish_max_threshold=max_thresholds, 
                                                bigfish_mean_threshold=mean_thresholds, bigfish_std_threshold=std_thresholds)
        
        return output

    def calculate_threshold(self, rna, voxel_size, spot_size, use_log_hook:bool =False):
        if use_log_hook:
            spot_radius_px = detection.get_object_radius_pixel(
                voxel_size_nm=voxel_size,
                object_radius_nm=spot_size,
                ndim=3)
            
            if self.verbose:
                print("spot radius (z axis): {:0.3f} pixels".format(spot_radius_px[0]))
                print("spot radius (yx plan): {:0.3f} pixels".format(spot_radius_px[-1]))

            spot_size = (spot_radius_px[0], spot_radius_px[-1], spot_radius_px[-1])
            spots, threshold = detection.detect_spots(
                images=rna,
                return_threshold=True,
                log_kernel_size=spot_size,
                minimum_distance=spot_size)
            
            if self.verbose:
                print("detected spots")
                print("\r shape: {0}".format(spots.shape))
                print("\r threshold: {0}".format(threshold))

            if self.step_output_dir is not None:
                plot.plot_elbow(
                    images=rna,
                    minimum_distance=spot_size,
                    log_kernel_size=spot_size,
                    title="Log Filter",
                    path_output=self.step_output_dir
                    )

        else:
            spot_size = spot_size
            spots, threshold = detection.detect_spots(
                images=rna,
                return_threshold=True,
                voxel_size=voxel_size,  # in nanometer (one value per dimension zyx)
                spot_radius=spot_size)  # in nanometer (one value per dimension zyx)
            if self.verbose:
                print("detected spots")
                print("\r shape: {0}".format(spots.shape))
                print("\r threshold: {0}".format(threshold))

        if self.step_output_dir is not None:
            plot.plot_elbow(
                images=rna,
                voxel_size=voxel_size,
                spot_radius=spot_size, 
                title="Normal Filter",
                path_output=self.step_output_dir
                )
        
        return threshold


class AutomaticThresholdingOutput(StepOutputsClass):
    def __init__(self, automatic_spot_detection_threshold) -> None:
        super().__init__()
        self.ModifyPipelineData = False
        self.automatic_spot_detection_threshold = automatic_spot_detection_threshold


class AutomaticSpotDetection(prePipelineStepsClass):
    """
    This uses list_images to calculate a threshold pre segmentation

   Inputs:
   PipelineSettings:
   - MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD: Lower Limit of images to use
   - MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD: Maximum number of images to use for spot detection
   - threshold_for_spot_detection: If you want to skip this step
   - CLUSTER_RADIUS: Cluster detection
   - minimum_spots_cluster: Minimum number of spots per cluster
   - use_log_filter_for_spot_detection: Use Log Filter for spot detection

   PipelineData:
   - list_images

   Experiment:
   - voxel_size_z
   - FISHChannel

   TerminatorScope:
   - voxel_size_yx
   - psf_z
   - psf_yx


   Returns:
   AutomaticThresholdingOutput (stored in pipelineData):
    threshold_for_spot_detection: average of the threshold detected from this step
    """

    def __init__(self) -> None:
        super().__init__()

    def main(self,
             list_images,
             voxel_size_z,
             voxel_size_yx,
             spot_z,
             spot_yx,
             FISHChannel,
             threshold_for_spot_detection,
             CLUSTER_RADIUS,
             minimum_spots_cluster,
             use_log_filter_for_spot_detection,
             MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD,
             MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD,
             **kwargs):

        # Luis's Code
        if (threshold_for_spot_detection is None) and (
                len(list_images) > MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD):
            number_images_to_test = np.min((MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD, len(list_images)))
            sub_section_images_to_test = list_images[:number_images_to_test]
            threshold_for_spot_detection = []
            for i in range(len(FISHChannel)):
                list_thresholds = []
                for _, image_selected in enumerate(sub_section_images_to_test):
                    threshold = BigFISH(image_selected,
                                        FISHChannel[i],
                                        voxel_size_z=voxel_size_z,
                                        voxel_size_yx=voxel_size_yx,
                                        psf_z=spot_z,
                                        psf_yx=spot_yx,
                                        cluster_radius=CLUSTER_RADIUS,
                                        minimum_spots_cluster=minimum_spots_cluster,
                                        use_log_filter_for_spot_detection=use_log_filter_for_spot_detection,
                                        threshold_for_spot_detection=None).detect()[2]
                    list_thresholds.append(threshold)
                # calculating the average threshold for all images removing min and max values.
                array_threshold_spot_detection = np.array(list_thresholds)
                min_val = np.min(array_threshold_spot_detection)
                max_val = np.max(array_threshold_spot_detection)
                mask_ts = (array_threshold_spot_detection != min_val) & (array_threshold_spot_detection != max_val)
                average_threshold_spot_detection = int(np.mean(array_threshold_spot_detection[mask_ts]))
                threshold_for_spot_detection.append(average_threshold_spot_detection)
            print(
                'Most images are noisy. An average threshold value for spot detection has been calculated using all images:',
                threshold_for_spot_detection)
        else:
            threshold_for_spot_detection = Utilities().create_list_thresholds_FISH(FISHChannel,
                                                                                   threshold_for_spot_detection)

        # outputs
        output = AutomaticThresholdingOutput(automatic_spot_detection_threshold=threshold_for_spot_detection)
        return output

#%% Trim Z Slices
class TrimZSlicesOutput(StepOutputsClass):
    def __init__(self, list_images, list_z_slices_per_image) -> None:
        super().__init__()
        self.ModifyPipelineData = True
        self.list_images = list_images
        self.list_z_slices_per_image = list_z_slices_per_image


class TrimZSlices(prePipelineStepsClass):
    """
    This step removes a user specified number of Z from each image

   Inputs:
   PipelineSettings:
   - MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE
   - NUMBER_Z_SLICES_TO_TRIM

   PipelineData:
   - list_images

   Experiment:
   - num_z_silces
   - number_of_images_to_process

   Returns:
   AutomaticThresholdingOutput (stored in pipelineData):
    - list_images (modifies PipelineData)
    - list_z_slices_per_image

    """

    def __init__(self) -> None:
        super().__init__()
        self.ModifyPipelineData = True

    def main(self,
             list_images,
             number_of_Z,
             num_img_2_run,
             MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE,
             NUMBER_Z_SLICES_TO_TRIM:int = 0,
             remove_z_slices_borders: bool = True,
             list_z_slices_per_image=None,
             list_selected_z_slices=None,
             **kwargs):

        if remove_z_slices_borders:
            list_images_trimmed = []
            if (remove_z_slices_borders and (list_selected_z_slices is None) and
                    (number_of_Z >= MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE)):
                list_selected_z_slices = np.arange(NUMBER_Z_SLICES_TO_TRIM, number_of_Z - NUMBER_Z_SLICES_TO_TRIM, 1)
            else:
                NUMBER_Z_SLICES_TO_TRIM = 0
                remove_z_slices_borders = False

            NUMBER_Z_SLICES_TO_TRIM = NUMBER_Z_SLICES_TO_TRIM
            if list_selected_z_slices is not None:
                for i in range(num_img_2_run):
                    if len(list_selected_z_slices) > list_images[i].shape[0]:
                        raise ValueError(
                            "Error: You are selecting z-slices that are outside the size of your image. In PipelineFISH, please use this option list_selected_z_slices=None ")
                    list_images_trimmed.append(list_images[i][list_selected_z_slices, :, :, :])
            # outputs
            output = TrimZSlicesOutput(list_images=list_images_trimmed, list_z_slices_per_image=list_selected_z_slices)
            return output
