import os
import inspect
import pickle

from . import Settings, ScopeClass, Experiment, DataContainer, PipelineOutputsClass, \
    PrePipelineOutputsClass, StepOutputsClass
from .Util.Utilities import Utilities


class Pipeline:
    def __init__(self,
                 settings: Settings,
                 scope: ScopeClass,
                 experiment: Experiment,
                 dataContainer: DataContainer,
                 independentSteps: list,
                 finalizationSteps: list,
                 sequentialSteps: list
                 ) -> None:
        # data storage classes
        self.settings = settings # user settings
        self.scope = scope # microscope params
        self.experiment = experiment # experiment params
        self.dataContainer = dataContainer # data storage

        # steps
        self.independentSteps = independentSteps
        self.finalizationSteps = finalizationSteps
        self.sequentialSteps = sequentialSteps

        self.experiment.pipeline_init()
        self.dataContainer.pipeline_init()
        self.settings.pipeline_init()
        self.scope.pipeline_init()

        self.check_requirements()

    def check_requirements(self):
        # inspects each steps main function to see if it has the required parameters
        no_default_params = []
        for step in self.independentSteps + self.finalizationSteps + self.sequentialSteps:
            step_func = step.main
            sig = inspect.signature(step_func)
            no_default_params.append([param.name for param in sig.parameters.values() if param.default is param.empty])

        # make the list of lists into a single list
        no_default_params = [item for sublist in no_default_params for item in sublist]

        # make the list unique
        no_default_params = list(set(no_default_params))

        print('The following parameters are required in the run the pipeline')
        print(no_default_params)

    def display_all_params(self):
        # inspects each steps main function to see if it has the required parameters
        all_params = []
        for step in self.independentSteps + self.finalizationSteps + self.sequentialSteps:
            step_func = step.run
            sig = inspect.signature(step_func)
            all_params.append([param.name for param in sig.parameters.values()])

        # make the list of lists into a single list
        all_params = [item for sublist in all_params for item in sublist]

        # make the list unique
        all_params = list(set(all_params))

        print('The following parameters are required in the run the pipeline')
        print(all_params)

    def execute_independent_steps(self):
        '''
        This method will run through all given prePipeline steps. PrePipeline steps are defined as steps that are ran on a subset of images.
        They can modify the pipelineData, they may also create new properties in pipelineData. they will also have the option to freeze the pipelineData in place

        '''
        for step in self.independentSteps:
            print(step)
            stepOutput = step.run(data=self.dataContainer,
                                  settings=self.settings,
                                  scope=self.scope,
                                  experiment=self.experiment)
            # check if should modify pipelineData
            if hasattr(stepOutput, 'ModifyPipelineData'):
                if stepOutput.ModifyPipelineData:
                    # find the attribute that both the pipelineData and stepOutput share
                    attrs = list(vars(stepOutput).keys()).copy()
                    for attr in attrs:
                        if hasattr(self.dataContainer, attr):
                            if getattr(stepOutput, attr) is not None:
                                print(f'Overwriting {attr} in pipelineData')
                                setattr(self.dataContainer, attr, getattr(stepOutput, attr))
                            else:
                                print(f'failed to modify {attr} because it was empty')
                            # remove that attribute from the stepOutput
                            delattr(stepOutput, attr)
            if stepOutput is None:
                print(f'{step} did not return a value')
            
            else:
                # append the attributes to prePipelineOutputs
                self.dataContainer.append(stepOutput)

    def execute_sequential_steps(self):
        for img_index in range(self.dataContainer.num_img_2_run):
            print('')
            print(' ###################### ')
            print('        IMAGE : ' + str(img_index))
            print(' ###################### ')
            print('    Image Name :  ', self.dataContainer.list_image_names[img_index])

            for step in self.sequentialSteps:
                singleImageSingleStepOutputs = step.run(id=img_index, data=self.dataContainer,
                                                        settings=self.settings,
                                                        scope=self.scope,
                                                        experiment=self.experiment)
                
                if singleImageSingleStepOutputs is None:
                    print(f'{step} did not return a value')
                else:
                    self.dataContainer.append(singleImageSingleStepOutputs)

    def execute_finalization_steps(self):
        for step in self.finalizationSteps:
            print(step)
            step.run(data=self.dataContainer,
                     settings=self.settings,
                     scope=self.scope,
                     experiment=self.experiment)

    def __post_init__(self):
        self.dataContainer.temp_folder_name = str('temp_results_' + self.experiment.initial_data_location.name)
        if not os.path.exists(self.dataContainer.temp_folder_name) and self.settings.save_files:
            os.makedirs(self.dataContainer.temp_folder_name)

    def run_up_to(self, step_name):
        all_steps = self.independentSteps + self.sequentialSteps + self.finalizationSteps
        if not (step_name in [step.__class__.__name__ for step in all_steps]):
            raise ValueError(f'{step_name} is not a valid step name')

        # remove all steps from step_name and onwards
        # check if its an independent step step and removes all after
        if step_name in [step.__class__.__name__ for step in self.independentSteps]:
            # get the index of the step
            index = [step.__class__.__name__ for step in self.independentSteps].index(step_name)
            # remove all steps after the index
            self.independentSteps = self.independentSteps[:index]
            self.execute_independent_steps()
            pickle.dump(self, open('pipeline.pkl', 'wb'))
        # check if its a sequential step and removes all after
        else: 
            self.execute_independent_steps()
            if step_name in [step.__class__.__name__ for step in self.sequentialSteps]:
                # get the index of the step
                index = [step.__class__.__name__ for step in self.sequentialSteps].index(step_name)
                # remove all steps after the index
                self.sequentialSteps = self.sequentialSteps[:index]
                self.execute_sequential_steps()
                pickle.dump(self, open('pipeline.pkl', 'wb'))
            # check if its a finalization step and removes all after
            else:
                self.execute_sequential_steps()
                if step_name in [step.__class__.__name__ for step in self.finalizationSteps]:
                    # get the index of the step
                    index = [step.__class__.__name__ for step in self.finalizationSteps].index(step_name)
                    # remove all steps after the index
                    self.finalizationSteps = self.finalizationSteps[:index]
                    self.execute_finalization_steps()
                    pickle.dump(self, open('pipeline.pkl', 'wb'))

    def run_single_step(self, step, modify_kwargs: dict = None):

        if modify_kwargs is not None:
            self.modify_kwargs(modify_kwargs)

        stepOutput = None

        if step.__class__.__base__.__name__ == 'SequentialStepsClass':
            for img_index in range(self.dataContainer.num_img_2_run):
                print('')
                print(' ###################### ')
                print('        IMAGE : ' + str(img_index))
                print(' ###################### ')
                print('    Image Name :  ', self.dataContainer.list_image_names[img_index])

                singleImgOutput = step.run(id=img_index, 
                                        data=self.dataContainer,
                                        settings=self.settings,
                                        scope=self.scope,
                                        experiment=self.experiment)
                
                if stepOutput is None:
                    stepOutput = singleImgOutput
                else:
                    stepOutput.append(singleImgOutput)

        else:
            print(step)
            stepOutput = step.run(data=self.dataContainer,
                                  settings=self.settings,
                                  scope=self.scope,
                                  experiment=self.experiment)
            
        return stepOutput
        
    def modify_kwargs(self, modify_kwargs: dict):
        kwargs_experiment = self.experiment.__dict__
        kwargs_scope = self.scope.__dict__
        kwargs_settings = self.settings.__dict__

        list_kwargs = [kwargs_experiment, kwargs_scope, kwargs_settings]

        for d in list_kwargs:
            for key, value in d.items():
                if key in modify_kwargs:
                    d[key] = modify_kwargs[key]

        self.experiment.__dict__ = kwargs_experiment
        self.scope.__dict__ = kwargs_scope
        self.settings.__dict__ = kwargs_settings



            