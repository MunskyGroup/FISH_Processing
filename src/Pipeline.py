import os
import inspect
import pickle
from abc import ABC, abstractmethod

from . import Settings, ScopeClass, Experiment, DataContainer, OutputClass, Parameters, StepClass
from .Util.Utilities import Utilities



class Pipeline:
    def __init__(self,
                 settings: Settings,
                 scope: ScopeClass,
                 experiment: Experiment,
                 ) -> None:
        self.Outputs = OutputClass() # final outputs
        self.settings = settings
        self.scope = scope
        self.experiment = experiment

    def check_requirements(self):
        self.get_parameters()

        Parameters.validate()

        # check if all required parameters are present
        params = Parameters.get_parameters()

        # check if all no default parameters are present
        for param in self.no_default_params:
            if param not in params:
                raise ValueError(f'{param} is required to run the pipeline')

    def display_all_params(self):
        required_params, all_params = StepClass.get_step_parameters()
        print('Required Parameters: ')
        for param in required_params:
            print(param)
        print('All Parameters: ')
        for param in all_params:
            print(param)

    def execute_independent_steps(self):
        '''
        This method will run through all given prePipeline steps. PrePipeline steps are defined as steps that are ran on a subset of images.
        They can modify the pipelineData, they may also create new properties in pipelineData. they will also have the option to freeze the pipelineData in place

        '''
        from src import IndependentStepsClass
        IndependentStepClass.execute()

    def execute_sequential_steps(self):
        from src import SequentialStepsClass
        SequentialStepClass.execute()

    def execute_finalization_steps(self):
        from src import FinalizationStepsClass
        FinalizationStepClass.execute()

    def __post_init__(self):
        self.check_requirements()

        self.save_location = self.DataContainer.save_location()
        
        Parameters.pipeline_init()

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
        Parameters.update_parameters(modify_kwargs)

    def get_step_parameters(self):
        self.no_default_params, self.all_params = StepClass.get_step_parameters()

class MultiPipeline:
    """
    Goal: To handle multiple pipelines and link them together to acheive more complicated tasks

    Given: 
    - A list of pipelines
    - A list of Datasets
    
    How:
    - Each pipeline will be run in sequence with all datasets
    - The calculated parameters will be saved
    - The prameters will be averaged and passed to the next pipeline in the sequence
    """

    def __init__(self, pipelines: list, datasets_locations: list):
        self.pipelines = pipelines
        self.datasets_locations = datasets_locations
        self.saved_results = {}
    

    def run(self):
        # run each pipeline with each dataset
        for p, pipeline in enumerate(self.pipelines):

            pipeline.clear_data()
            for dataset_loc in self.datasets_locations:
                pipeline.set_dataset(dataset_loc)
                pipeline.execute_independent_steps()
                pipeline.execute_sequential_steps()
                pipeline.execute_finalization_steps()
                pipeline.save_outputs()
            
            self.average_parameters(pipeline)

            self.load_results(self.pipelines[p+1])





class DataCatastaphous:
    """
    Goal: To handle multiple pipelines and link them together to acheive more complicated tasks

    How:
    - Each pipeline will be run in sequence
    - The outputs of each pipeline will be stored in a dictionary of location
    - Parameters from previous steps will be passed to the next pipeline
    - 
    
    """
    def __init__(self):
        pass




            