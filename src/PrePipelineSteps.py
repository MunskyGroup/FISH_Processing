import numpy as np


from . import PipelineDataClass, PipelineSettings, Experiment, ScopeClass, prePipelineStepsClass, StepOutputsClass

from src.Util.Utilities import Utilities
from src.Util.BigFish import BigFISH

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
        self.list_images = None

    def load_in_attributes(self):
        self.list_images = self.pipelineData.list_images

    def main(self):
        # Luis's code
        if len(self.list_images[0].shape) < 3:
            list_images_extended = [np.expand_dims(img, axis=0) for img in self.list_images ]
            self.list_images = list_images_extended
            
        if len(self.list_images[0].shape) < 4:
            list_images_extended = [np.expand_dims(img, axis=0) for img in self.pipelineData.list_images ]
            self.list_images = list_images_extended
        else:
            self.list_images = self.list_images

        # outputs
        output = ConsolidateImageShapesOutput(list_images=self.list_images)
        return output


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
        self.channels_with_FISH = None
        self.sharpness_threshold = None
        self.list_images = None
        self.remove_out_of_focus_images = None
        self.ModifyPipelineData = True

    def load_in_attributes(self):
        # Required Inputs:
        self.remove_out_of_focus_images = self.pipelineSettings.remove_out_of_focus_images
        self.sharpness_threshold = self.pipelineSettings.sharpness_threshold
        self.list_images = self.pipelineData.list_images
        self.channels_with_FISH = self.experiment.FISHChannel


    def main(self):
        # Modified Luis's Code
        # this can be rewritten to be more efficient, calculate sharpness is ran either way, we just change the outputs
        if self.remove_out_of_focus_images:
            list_metric_sharpness_images, list_is_image_sharp, list_sharp_images = Utilities().calculate_sharpness(
                self.list_images, channels_with_FISH=self.channels_with_FISH,threshold=self.sharpness_threshold)
        else:
            list_is_image_sharp = np.ones(len(self.list_images))
            list_is_image_sharp = [bool(x) for x in list_is_image_sharp]
            list_metric_sharpness_images = Utilities().calculate_sharpness(self.list_images,
                                                                           channels_with_FISH=self.channels_with_FISH,
                                                                           threshold=self.sharpness_threshold)[0]
            list_sharp_images = None

        # outputs
        output = CalculateSharpnessOutput(list_is_image_sharp=list_is_image_sharp, list_images=list_sharp_images,
                                          list_metric_sharpness_images=list_metric_sharpness_images)

        return output


class AutomaticThresholdingOutput(StepOutputsClass):
    def __init__(self, threshold_for_spot_detection) -> None:
        super().__init__()
        self.ModifyPipelineData = False
        self.threshold_for_spot_detection = threshold_for_spot_detection


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
        self.use_log_filter_for_spot_detection = None
        self.minimum_spots_cluster = None
        self.CLUSTER_RADIUS = None
        self.threshold_for_spot_detection = None
        self.channels_with_FISH = None
        self.psf_yx = None
        self.psf_z = None
        self.voxel_size_yx = None
        self.voxel_size_z = None
        self.list_sharp_images = None
        self.MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = None
        self.MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = None
        self.ModifyPipelineData = False

    def load_in_attributes(self):
        self.MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = (
            self.pipelineSettings.MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD)
        self.MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = (
            self.pipelineSettings.MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD)
        self.list_sharp_images = self.pipelineData.list_images
        self.voxel_size_z = self.experiment.voxel_size_z
        self.voxel_size_yx = self.terminatorScope.voxel_size_yx
        self.psf_z = self.terminatorScope.psf_z
        self.psf_yx = self.terminatorScope.psf_yx
        self.channels_with_FISH = self.experiment.FISHChannel
        self.threshold_for_spot_detection = self.pipelineSettings.threshold_for_spot_detection
        self.CLUSTER_RADIUS = self.pipelineSettings.CLUSTER_RADIUS
        self.minimum_spots_cluster = self.pipelineSettings.minimum_spots_cluster
        self.use_log_filter_for_spot_detection = self.pipelineSettings.use_log_filter_for_spot_detection


    def main(self):
        # Luis's Code
        if (self.threshold_for_spot_detection is None) and (len(self.list_sharp_images) > self.MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD):
            number_images_to_test = np.min((self.MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD, len(self.list_sharp_images)))
            sub_section_images_to_test = self.list_sharp_images[:number_images_to_test]
            threshold_for_spot_detection = []
            for i in range(len(self.channels_with_FISH)):
                list_thresholds = []
                for _, image_selected in enumerate(sub_section_images_to_test):
                    threshold = BigFISH(image_selected,
                                        self.channels_with_FISH[i],
                                        voxel_size_z=self.voxel_size_z,
                                        voxel_size_yx=self.voxel_size_yx,
                                        psf_z=self.psf_z,
                                        psf_yx=self.psf_yx,
                                        cluster_radius=self.CLUSTER_RADIUS,
                                        minimum_spots_cluster=self.minimum_spots_cluster,
                                        use_log_filter_for_spot_detection=self.use_log_filter_for_spot_detection,
                                        threshold_for_spot_detection=None).detect()[2]
                    list_thresholds.append(threshold)
                # calculating the average threshold for all images removing min and max values.
                array_threshold_spot_detection = np.array(list_thresholds)
                min_val = np.min(array_threshold_spot_detection)
                max_val = np.max(array_threshold_spot_detection)
                mask_ts = (array_threshold_spot_detection != min_val) & (array_threshold_spot_detection != max_val)
                average_threshold_spot_detection = int(np.mean(array_threshold_spot_detection[mask_ts]))
                threshold_for_spot_detection.append(average_threshold_spot_detection)
            print('Most images are noisy. An average threshold value for spot detection has been calculated using all images:', threshold_for_spot_detection)
        else:
            threshold_for_spot_detection = Utilities().create_list_thresholds_FISH(self.channels_with_FISH, self.threshold_for_spot_detection)

        # outputs
        output = AutomaticThresholdingOutput(threshold_for_spot_detection=threshold_for_spot_detection)
        return output


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

    def load_in_attributes(self):
        self.MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE = self.pipelineSettings.MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE # This constant is only used to remove the extre z-slices on the original image.
        self.num_z_slices = self.experiment.number_of_Z
        self.list_images = self.pipelineData.list_images
        self.NUMBER_Z_SLICES_TO_TRIM = self.pipelineSettings.NUMBER_Z_SLICES_TO_TRIM # This constant indicates the number of z_slices to remove from the border.
        self.number_images = self.experiment.number_of_images_to_process

    def main(self):
        # this is stupid but whatever
        if self.pipelineSettings.remove_z_slices_borders:
            self.remove_z_slices_borders = True
        else:
            print('Trim Z Slices will not be ran as the flag is not set')
        
        if hasattr(self.pipelineData, 'list_z_slices_per_image'):
            list_selected_z_slices = self.pipelineData.list_z_slices_per_image
        else:
            list_selected_z_slices = None

        if self.remove_z_slices_borders:
            list_images_trimmed = []
            if (self.remove_z_slices_borders and (list_selected_z_slices is None) and
                    (self.num_z_slices >= self.MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE)):
                list_selected_z_slices = np.arange(self.NUMBER_Z_SLICES_TO_TRIM, self.num_z_slices - self.NUMBER_Z_SLICES_TO_TRIM, 1)
                self.remove_z_slices_borders = True
            else:
                self.NUMBER_Z_SLICES_TO_TRIM = 0
                self.remove_z_slices_borders = False
            self.NUMBER_Z_SLICES_TO_TRIM = self.NUMBER_Z_SLICES_TO_TRIM
            if not (list_selected_z_slices is None):
                for i in range(self.number_images):
                    if len(list_selected_z_slices) > self.list_images[i].shape[0]:
                        raise ValueError("Error: You are selecting z-slices that are outside the size of your image. In PipelineFISH, please use this option list_selected_z_slices=None ")
                    list_images_trimmed.append(self.list_images[i][list_selected_z_slices,:,:,:])
            # outputs
            output = TrimZSlicesOutput(list_images=list_images_trimmed, list_z_slices_per_image=list_selected_z_slices)
            return output





