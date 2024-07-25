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
    def __init__(self) -> None:
        super().__init__()
        self.ModifyPipelineData = True

    def main(self, pipelineData:PipelineDataClass):
        self.pipelineData = pipelineData
        # inputs
        list_images = self.pipelineData.list_images

        # Luis's code
        if len(list_images[0].shape) < 3:
            list_images_extended = [ np.expand_dims(img,axis=0) for img in list_images ] 
            list_images = list_images_extended
            
        if len(list_images[0].shape) < 4:
            list_images_extended = [ np.expand_dims(img,axis=0) for img in self.pipelineData.list_images ] 
            list_images = list_images_extended
        else:
            list_images = list_images

        # outputs
        output = ConsolidateImageShapesOutput(list_images=list_images)
        # self.pipelineData.list_images = list_images

        return output
    
    def run(self, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, terminatorScope:ScopeClass, experiment:Experiment):
        return self.main(pipelineData)


class CalculateSharpnessOutput(StepOutputsClass):
    def __init__(self, list_is_image_sharp, list_images, list_metric_sharpeness_images) -> None:
        super().__init__()
        self.ModifyPipelineData = True
        self.list_is_image_sharp = list_is_image_sharp
        self.list_images = list_images
        self.list_metric_sharpeness_images = list_metric_sharpeness_images

class CalculateSharpness(prePipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()
        self.ModifyPipelineData = True



    def main(self, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, experiment:Experiment):
        self.pipelineData = pipelineData
        self.pipelineSettings = pipelineSettings
        self.experiment = experiment
        # inputs
        remove_out_of_focus_images = self.pipelineSettings.remove_out_of_focus_images
        list_images = self.pipelineData.list_images
        sharpness_threshold = self.pipelineSettings.sharpness_threshold
        channels_with_FISH = self.experiment.FISHChannel


        # Modified Luis's Code 
        if remove_out_of_focus_images == True: # this can be rewritten to be more efficient, calculate sharpness is ran either way, we just change the outputs
            list_metric_sharpeness_images, list_is_image_sharp,list_sharp_images = Utilities().calculate_sharpness(list_images, channels_with_FISH=channels_with_FISH,threshold=sharpness_threshold)
        else:
            list_is_image_sharp = np.ones(len(list_images))
            list_is_image_sharp = [bool(x) for x in list_is_image_sharp]
            list_metric_sharpeness_images = Utilities().calculate_sharpness(list_images, channels_with_FISH=channels_with_FISH,threshold=sharpness_threshold)[0]
            list_sharp_images = None


        # outputs
        output = CalculateSharpnessOutput(list_is_image_sharp=list_is_image_sharp, list_images=list_sharp_images, list_metric_sharpeness_images=list_metric_sharpeness_images)
        # self.pipelineData.list_is_image_sharp = list_is_image_sharp
        # self.pipelineData.list_metric_sharpeness_images = list_metric_sharpeness_images
        # self.pipelineData.list_images = list_sharp_images

        return output
    
    def run(self, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, terminatorScope:ScopeClass, experiment:Experiment):
        return self.main(pipelineData=pipelineData, experiment=experiment, pipelineSettings=pipelineSettings)


class AutomaticThresholdingOutput(StepOutputsClass):
    def __init__(self, threshold_for_spot_detection) -> None:
        super().__init__()
        self.ModifyPipelineData = False
        self.threshold_for_spot_detection = threshold_for_spot_detection

class AutomaticSpotDetection(prePipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()
        self.ModifyPipelineData = False


    def main(self,pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, experiment:Experiment, terminatorScope:ScopeClass):
        self.pipelineData = pipelineData
        self.pipelineSettings = pipelineSettings
        self.experiment = experiment
        self.scope = terminatorScope
        # inputs
        MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = pipelineSettings.MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD
        MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = pipelineSettings.MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD

        list_sharp_images = self.pipelineData.list_images
        voxel_size_z = self.experiment.voxel_size_z
        voxel_size_yx = self.scope.voxel_size_yx
        psf_z = self.scope.psf_z
        psf_yx = self.scope.psf_yx
        channels_with_FISH = self.experiment.FISHChannel
        threshold_for_spot_detection = self.pipelineSettings.threshold_for_spot_detection
        CLUSTER_RADIUS = self.pipelineSettings.CLUSTER_RADIUS
        minimum_spots_cluster = self.pipelineSettings.minimum_spots_cluster
        use_log_filter_for_spot_detection = self.pipelineSettings.use_log_filter_for_spot_detection


        # Luis's Code
        if (threshold_for_spot_detection == None) and (len(list_sharp_images)>MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD):
            number_images_to_test = np.min((MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD,len(list_sharp_images)))
            sub_section_images_to_test =list_sharp_images[:number_images_to_test]
            threshold_for_spot_detection =[]
            for i in range(len(channels_with_FISH)):
                list_thresholds=[]

                for _, image_selected in enumerate(sub_section_images_to_test):
                    threshold = BigFISH(image_selected,
                                        channels_with_FISH[i], 
                                        voxel_size_z = voxel_size_z,
                                        voxel_size_yx = voxel_size_yx, 
                                        psf_z = psf_z, 
                                        psf_yx = psf_yx, 
                                        cluster_radius=CLUSTER_RADIUS,
                                        minimum_spots_cluster=minimum_spots_cluster, 
                                        use_log_filter_for_spot_detection =use_log_filter_for_spot_detection,
                                        threshold_for_spot_detection=None).detect()[2]
                    list_thresholds.append(threshold)
                # calculating the average threshold for all images removing min and max values.
                array_threshold_spot_detection = np.array(list_thresholds)
                min_val = np.min(array_threshold_spot_detection)
                max_val = np.max(array_threshold_spot_detection)
                mask_ts = (array_threshold_spot_detection != min_val) & (array_threshold_spot_detection != max_val)
                average_threshold_spot_detection= int(np.mean(array_threshold_spot_detection[mask_ts]))
                threshold_for_spot_detection.append(average_threshold_spot_detection)
            print('Most images are noisy. An average threshold value for spot detection has been calculated using all images:', threshold_for_spot_detection)
        else:
            threshold_for_spot_detection = Utilities().create_list_thresholds_FISH(channels_with_FISH,threshold_for_spot_detection)


        # outputs
        output = AutomaticThresholdingOutput(threshold_for_spot_detection=threshold_for_spot_detection)
        # self.pipelineData.threshold_for_spot_detection = threshold_for_spot_detection

        return output

    def run(self, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, terminatorScope:ScopeClass, experiment:Experiment):
        return self.main(pipelineData=pipelineData, experiment=experiment, pipelineSettings=pipelineSettings, terminatorScope=terminatorScope)


class TrimZSlicesOutput(StepOutputsClass):
    def __init__(self, list_images, list_z_slices_per_image) -> None:
        super().__init__()
        self.ModifyPipelineData = True
        self.list_images = list_images
        self.list_z_slices_per_image = list_z_slices_per_image

class TrimZSlices(prePipelineStepsClass):
    def __init__(self) -> None:
        super().__init__()
        self.ModifyPipelineData = True



    def main(self, pipelineData:PipelineDataClass, experiment:Experiment, pipelineSettings:PipelineSettings):
        self.pipelineData = pipelineData
        self.experiment = experiment
        self.pipelineSettings = pipelineSettings
        # inputs
        MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE = self.pipelineSettings.MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE # This constant is only used to remove the extre z-slices on the original image.
        num_z_silces = self.experiment.number_of_Z
        list_images = self.pipelineData.list_images
        NUMBER_Z_SLICES_TO_TRIM = self.pipelineSettings.NUMBER_Z_SLICES_TO_TRIM # This constant indicates the number of z_slices to remove from the border.
        number_images = self.experiment.number_of_images_to_process

        # this is stupid but whatever
        if self.pipelineSettings.remove_z_slices_borders:
            self.remove_z_slices_borders = True
        else:
            print('Trim Z Slices will not be ran as the flag is not set')
        
        if hasattr(self.pipelineData, 'list_z_slices_per_image'):
            list_selected_z_slices = self.pipelineData.list_z_slices_per_image
        else:
            list_selected_z_slices = None

                    # Modified Luis's code
        if self.remove_z_slices_borders:
            # if self.pipelineData.list_z_slices_per_image is None:
            #     list_selected_z_slices = None
            # else:
            #     list_selected_z_slices = self.pipelineData.list_z_slices_per_image)

            list_images_trimmed = []
            if (self.remove_z_slices_borders == True) and  (list_selected_z_slices is None) and (num_z_silces>=MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE): 
                list_selected_z_slices = np.arange(NUMBER_Z_SLICES_TO_TRIM,num_z_silces-NUMBER_Z_SLICES_TO_TRIM,1)
                self.remove_z_slices_borders = True
            else:
                NUMBER_Z_SLICES_TO_TRIM = 0
                self.remove_z_slices_borders = False
            self.NUMBER_Z_SLICES_TO_TRIM = NUMBER_Z_SLICES_TO_TRIM
            if not (list_selected_z_slices is None):
                for i in range (number_images):
                    if len(list_selected_z_slices) > list_images[i].shape[0]:
                        raise ValueError("Error: You are selecting z-slices that are outside the size of your image. In PipelineFISH, please use this option list_selected_z_slices=None ")
                    list_images_trimmed.append(list_images[i][list_selected_z_slices,:,:,:])
            # outputs
            output = TrimZSlicesOutput(list_images=list_images_trimmed, list_z_slices_per_image=list_selected_z_slices)
            # self.pipelineData.list_images = list_images_trimmed
            # self.pipelineData.list_z_slices_per_image = [img.shape[0] for img in self.pipelineData.list_images] # number of z-slices in the figure
        return output
    
    def run(self, pipelineData:PipelineDataClass, pipelineSettings:PipelineSettings, terminatorScope:ScopeClass, experiment:Experiment):
        return self.main(pipelineData=pipelineData, experiment=experiment, pipelineSettings=pipelineSettings)