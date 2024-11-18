import os
import inspect
import pickle
from abc import ABC, abstractmethod

from . import Settings, ScopeClass, Experiment, DataContainer, OutputClass, Parameters, StepClass
from .Util.Utilities import Utilities



class Pipeline:
    def __init__(self,
                 ) -> None:
        pass

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
        from src.GeneralStep import IndependentStepClass
        IndependentStepClass().execute()

    def execute_sequential_steps(self):
        from src.GeneralStep import SequentialStepsClass
        SequentialStepsClass().execute()

    def execute_finalization_steps(self):
        from src.GeneralStep import FinalizingStepClass
        FinalizingStepClass().execute()

    def __post_init__(self):
        self.check_requirements()

        self.save_location = self.DataContainer.save_location()
        
        Parameters.pipeline_init()
        
    def modify_kwargs(self, modify_kwargs: dict):
        Parameters.update_parameters(modify_kwargs)

    def get_step_parameters(self):
        self.no_default_params, self.all_params = StepClass.get_step_parameters()

    def get_independent_steps(self):
        from src.GeneralStep import IndependentStepClass
        self.independent_steps = IndependentStepClass._instances
        return self.independent_steps

    def get_sequential_steps(self):
        from src.GeneralStep import SequentialStepsClass
        self.sequential_steps = SequentialStepsClass._instances
        return self.sequential_steps
    
    def get_finalization_steps(self):
        from src.GeneralStep import FinalizingStepClass
        self.finalization_steps = FinalizingStepClass._instances
        return self.finalization_steps

    def run(self):
        self.execute_independent_steps()
        self.execute_sequential_steps()
        self.execute_finalization_steps()


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




            