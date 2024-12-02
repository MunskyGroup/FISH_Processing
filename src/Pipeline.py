import os
import inspect
import pickle
from typing import Union
from abc import ABC, abstractmethod
from itertools import cycle, islice
import json

from . import OutputClass,  StepClass
from .Parameters import Parameters, Experiment, Settings, ScopeClass, DataContainer
from .Util.Utilities import Utilities



class Pipeline:
    def __init__(self,
                    experiment_location: Union[str, list[str], list[list[str]]] = None,
                    parameters: Union[dict, list[dict]] = None,
                    steps: Union[list[str], list[list[str]]] = None,
                 ) -> None:
        self.experiment_location = experiment_location 
        self.parameters = parameters
        self.steps = steps

    def check_requirements(self):
        self.get_step_parameters()

        Parameters.validate()

        # check if all required parameters are present
        params = Parameters.get_parameters()

        params_to_ignore = ['nuc_mask', 'cell_mask', 'masks', 'images', 'image', 'fov', 'timepoint']

        # check if all no default parameters are present
        for param in self.no_default_params:
            if param not in params and param not in params_to_ignore:
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
        def zip_expand(*iterables):
            # Find the length of the longest iterable
            max_length = max(len(iterable) for iterable in iterables)

            # check if all iterables have the same length or if they have length 1
            if all(len(iterable) == max_length or len(iterable) == 1 for iterable in iterables):
            
                # Create a new list of iterables where any iterable with length 1 is expanded to match max_length
                expanded_iterables = [
                    iterable if len(iterable) > 1 else list(islice(cycle(iterable), max_length))
                    for iterable in iterables
                ]
                
                # Use zip to combine the expanded iterables
                return zip(*expanded_iterables)
            else:
                raise ValueError('All iterables must have the same length or length 1')

        if self.experiment_location is None:
            self.experiment_location = [Experiment().initial_data_location]
        if self.parameters is None:
            self.parameters = [Parameters.get_parameters()]
        if self.steps is None:
            self.steps = [[*[i.__class__.__name__ for i in self.get_independent_steps()], *[i.__class__.__name__ for i in self.get_sequential_steps()],
                *[i.__class__.__name__ for i in self.get_finalization_steps()]]]
        
        for locations, params, steps in zip_expand(self.experiment_location, self.parameters, self.steps):
            Parameters.initialize_parameters_instances()
            self.modify_kwargs(params)
            self._run(locations, steps)

    def run_on_cluster(self, remote_folder, name: str = None):
        self.save_pipeline(name=Settings().name if name is None else name)
        self.send_pipeline_to_cluster(remote_folder)

    def _run(self, locations, steps):
        # save locations and steps
        if steps is not None:
            StepClass.initalize_steps_from_list(steps)
        if locations is not None:
            Experiment().initial_data_location = locations
        # method to to execute the steps in order
        self.check_requirements()
        print('Running Independent Steps')
        self.execute_independent_steps()
        print('Running Sequential Steps')
        self.execute_sequential_steps()
        print('Running Finalization Steps')
        self.execute_finalization_steps()
        self.clear_pipeline()


    def save_pipeline(self, name: str):
        # save params as a dictionary
        if self.parameters is None:
            params = [Parameters.get_parameters()]

        # save save steps as a dictionary
        if self.steps is None:
            steps = [[*[i.__class__.__name__ for i in self.get_independent_steps()], *[i.__class__.__name__ for i in self.get_sequential_steps()],
                    *[i.__class__.__name__ for i in self.get_finalization_steps()]]]
        
        # save these as a dictionary
        pipeline = {'params': params, 'steps': steps}

        # save the pipeline as txt file
        file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        parent_dir = os.path.dirname(file_path)

        pipeline_dir = os.path.join(parent_dir, 'Pipelines')
        
        self.pipeline_dictionary_location = os.path.join(pipeline_dir, f'{name}.txt')
        with open(self.pipeline_dictionary_location, 'w') as f:
            json.dump(pipeline, f)

    def send_pipeline_to_cluster(self, remote_folder):
        from .Send_To_Cluster import run_on_cluster
        run_on_cluster(remote_folder , self.pipeline_dictionary_location) 

    def clear_pipeline(self):
        Parameters.clear_instances()
        StepClass.clear_instances()

if __name__ == '__main__':
    # get the current file path
    file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print (file_path)