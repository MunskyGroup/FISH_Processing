import os

from . import PipelineSettings, ScopeClass, Experiment, PipelineDataClass, PipelineOutputsClass, \
    PrePipelineOutputsClass, StepOutputsClass
from .Util.Utilities import Utilities


# from .GeneralOutputClasses import PipelineOutputsClass
# from .PipelineSettings import PipelineSettings
# from .MicroscopeClass import ScopeClass
# from .ExperimentClass import Experiment
# from .PipelineDataClass import PipelineDataClass

class Pipeline:
    def __init__(self,
                 pipelineSettings: PipelineSettings,
                 terminatorScope: ScopeClass,
                 experiment: Experiment,
                 pipelineData: PipelineDataClass,
                 prePipelineSteps: list,
                 postPipelineSteps: list,
                 pipelineSteps: list
                 ) -> None:
        self.pipelineSettings = pipelineSettings
        self.terminatorScope = terminatorScope
        self.experiment = experiment
        self.pipelineData = pipelineData
        self.prePipelineSteps = prePipelineSteps
        self.postPipelineSteps = postPipelineSteps
        self.pipelineSteps = pipelineSteps
        self.pipelineData.num_img_2_run = min(self.pipelineSettings.user_select_number_of_images_to_run,
                                              self.experiment.number_of_images_to_process)

        self.pipelineData.temp_folder_name = str('temp_results_' + self.experiment.initial_data_location.name)
        if not os.path.exists(self.pipelineData.temp_folder_name) and self.pipelineSettings.save_files:
            os.makedirs(self.pipelineData.temp_folder_name)

        if self.pipelineSettings.save_all_images:
            self.pipelineData.filtered_folder_name = str(
                'filtered_images_' + self.experiment.initial_data_location.name)
            if not os.path.exists(self.pipelineData.filtered_folder_name):
                os.makedirs(self.pipelineData.filtered_folder_name)

        self.override_bad_settings()

    def override_bad_settings(self):
        '''if np.min(self.pipelineData.list_z_slices_per_image) < 5:
            self.pipelineSettings.optimization_segmentation_method = 'center_slice' # MAYBE KEEP THIS
            print(f'You picked a shit optimazation setting for segmentation because you have {self.pipelineData.list_z_slices_per_image} z slices \n',
                  'I will pick a better one for you: center_slice')'''
        pass

    def run_pre_pipeline_steps(self):
        '''
        This method will run through all given prePipeline steps. PrePipeline steps are defined as steps that are ran on a subset of images.
        They can modify the pipelineData, they may also create new properties in pipelineData. they will also have the option to freeze the pipelineData in place

        '''
        for step in self.prePipelineSteps:
            print(step)
            stepOutput = step.run(pipelineData=self.pipelineData,
                                  pipelineSettings=self.pipelineSettings,
                                  terminatorScope=self.terminatorScope,
                                  experiment=self.experiment)
            if stepOutput is None:
                raise ValueError(f'{step} did not return a value')
            if hasattr(stepOutput, 'ModifyPipelineData'):
                if stepOutput.ModifyPipelineData:
                    # find the attribute that both the pipelineData and stepOutput share
                    attrs = list(vars(stepOutput).keys()).copy()
                    for attr in attrs:
                        if hasattr(self.pipelineData, attr):
                            if getattr(stepOutput, attr) is not None and len(getattr(stepOutput, attr)) != 0:
                                print(f'Overwriting {attr} in pipelineData')
                                setattr(self.pipelineData, attr, getattr(stepOutput, attr))
                            else:
                                print(f'failed to modify {attr} because it was empty')
                            # remove that attribute from the stepOutput
                            delattr(stepOutput, attr)
                # append the attributes to prePipelineOutputs
            self.pipelineData.append(stepOutput)

    def run_pipeline_steps(self):
        for img_index in range(self.pipelineData.num_img_2_run):
            print('')
            print(' ###################### ')
            print('        IMAGE : ' + str(img_index))
            print(' ###################### ')
            print('    Image Name :  ', self.pipelineData.list_image_names[img_index])

            img = self.pipelineData.list_images[img_index]
            img_shape = list(img.shape)
            if self.pipelineSettings.remove_z_slices_borders:
                img_shape = list(img.shape)
                img_shape[0] = img_shape[0] + 2 * self.pipelineSettings.NUMBER_Z_SLICES_TO_TRIM
                print('    Original Image Shape :                    ', img_shape)
                print('    Trimmed z_slices at each border :        ', self.pipelineSettings.NUMBER_Z_SLICES_TO_TRIM)
            else:
                print('    Original Image Shape :                   ', img_shape)
                print('    Image sharpness metric :                 ',
                      self.pipelineData.CalculateSharpnessOutput.list_is_image_sharp[id])

            for step in self.pipelineSteps:
                singleImageSingleStepOutputs = step.run(id=img_index, pipelineData=self.pipelineData,
                                                        pipelineSettings=self.pipelineSettings,
                                                        terminatorScope=self.terminatorScope,
                                                        experiment=self.experiment)
                if singleImageSingleStepOutputs is None:
                    raise ValueError(f'{step} did not return a value')
                self.pipelineData.append(singleImageSingleStepOutputs)

    def run_post_pipeline_steps(self):
        for step in self.postPipelineSteps:
            print(step)
            step.run(pipelineData=self.pipelineData,
                     pipelineSettings=self.pipelineSettings,
                     terminatorScope=self.terminatorScope,
                     experiment=self.experiment)
