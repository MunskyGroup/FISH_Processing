import os
import inspect
import pickle
from typing import Union
from abc import ABC, abstractmethod

from . import OutputClass,  StepClass
from .Parameters import Parameters, Experiment, Settings, ScopeClass, DataContainer
from .Util.Utilities import Utilities



class Pipeline:
    def __init__(self,
                    experiment_location: Union[str, list[str]] = None,
                 ) -> None:
        self.experiment_location = experiment_location

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
        if self.experiment_location is None: # first case: no experiment location is given in pipeline
            if Experiment().initial_data_location is None: # if experiment location is not set
                raise ValueError('Experiment location is not set')
            else: # if experiment location is set
                self._run()
        else: # second case: experiment location is given in pipeline
            if type(self.experiment_location) == list: # if multiple experiment locations are given
                for f in self.experiment_location:
                    Experiment().initial_data_location = f
                    self._run()
            else: # if only one experiment location is given
                Experiment().initial_data_location = self.experiment_location
                self._run()

    def run_on_cluster(self):
        self.save_pipeline(name=Settings().name)
        self.send_pipeline_to_cluster()

    def _run(self):
        # method to to execute the steps in order
        self.check_requirements()
        self.execute_independent_steps()
        self.execute_sequential_steps()
        self.execute_finalization_steps()

    def save_pipeline(self, name: str):
        # save params as a dictionary
        params = Parameters.get_parameters()

        # save save steps as a dictionary
        steps = {'independent_steps': [i.__class__.__name__ for i in self.get_independent_steps()],
                 'sequential_steps': [i.__class__.__name__ for i in self.get_sequential_steps()],
                 'finalization_steps': [i.__class__.__name__ for i in self.get_finalization_steps()]}
        
        # save these as a dictionary
        pipeline = {'params': params, 'steps': steps}

        # save the pipeline as txt file
        file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        parent_dir = os.path.dirname(file_path)

        pipeline_dir = os.path.join(parent_dir, 'Pipelines')
        
        with open(os.path.join(pipeline_dir, f'{name}.txt'), 'wb') as f:
            pickle.dump(pipeline, f)

    def send_pipeline_to_cluster(self):
        pass











if __name__ == '__main__':
    # get the current file path
    file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print (file_path)