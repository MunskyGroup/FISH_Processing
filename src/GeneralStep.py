import os
from abc import ABC, abstractmethod
import inspect

from .Parameters import Parameters
# from . import Settings, Experiment, ScopeClass, DataContainer


class StepClass(ABC):
    _instances = []

    def __init__(self):
        StepClass._instances.append(self)

    @classmethod
    def get_all_instances(cls):
        # Class method to return all instances of the parent class
        return cls._instance

    @classmethod
    def get_all_parameters(cls):
        # inspects each steps main function to see if it has the required parameters
        steps = cls.get_all_instances()
        no_default_params = []
        all_params = []
        for step in steps:
            step_func = step.main
            sig = inspect.signature(step_func)
            no_default_params.append([param.name for param in sig.parameters.values() if param.default is param.empty])
            all_params.append([param.name for param in sig.parameters.values()])

        return list(set(no_default_params)), list(set(all_params))
    
    def check_setting_requirements(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def load_in_attributes(self, id: int = None):
        """
        This is where the magic happens. This function will load in all the attributes of the class and return them as a dictionary.
        This allows all the bs that I decided to force on my code to not matter, and user can just write whatever they want in the main functions
        As long as the attributes are unique and saved using a step output class, this function will load them in.
        """

        params = Parameters.get_parameters()
        

        if id is not None: # TODO This will need to be changed for Dask arrays
            params[id] =  id
            params['image'] = self.data.list_images[id]
            params['image_name'] = os.path.splitext(self.data.list_image_names[id])[0]
            try:
                params['cell_mask'] = self.data.masks_complete_cells[id]
            except AttributeError:
                params['cell_mask'] = None
            try:
                params['nuc_mask'] = self.data.masks_nuclei[id]
            except AttributeError:
                params['nuc_mask'] = None
            try:
                params['cyto_mask'] = self.data.masks_cytosol[id]
            except AttributeError:
                params['cyto_mask'] = None
        
        print(params)
        return params
    
    def create_step_output_dir(self, output_location = None, **kwargs):
        if output_location is not None:
            self.step_output_dir = os.path.join(output_location, self.__class__.__name__)
            os.makedirs(self.step_output_dir, exist_ok=True)
        else:
            self.step_output_dir = None

    @abstractmethod
    def main(self, **kwargs):
        pass

    def run(self, id: int = None):
        kwargs = self.load_in_attributes(id)
        return self.main(**kwargs) 

class SequentialStepsClass(StepClass):
    _instances = []
    def __init__(self):
        super().__init__()
        SequentialStepsClass._instances.append(self)
        self.is_first_run = True

    def execute(self):
        self.num_chunks_to_run = Parameters.get_parameters()['num_chunks_to_run']
        for id in range(self.num_chunks_to_run):
            for step in SequentialStepsClass._instances:
                print('++++++++++++++++++++++++++++')
                print('Running : ', step)
                print('++++++++++++++++++++++++++++')
                step.run(id)

    def run(self, id: int = None):
        self.num_chunks_to_run = Parameters.get_parameters()['num_chunks_to_run']
        if id is None:  # allows for pipelineSteps to be run a pre or postPipeline
            for id in range(self.num_chunks_to_run):
                print('')
                print(' ###################### ')
                print('        IMAGE : ' + str(id))
                print(' ###################### ')
                params = self.load_in_attributes(id)
                self.create_step_output_dir(**params)
                self.on_first_run(id)
                output = self.main(**params)
        else:
            print('')
            print(' ###################### ')
            print('        IMAGE : ' + str(id))
            print(' ###################### ')
            params = self.load_in_attributes(id)
            self.create_step_output_dir(**params)
            self.on_first_run(id)
            output = self.main(**params)
        
        return output

    def main(self, **kwargs):
        pass

    def on_first_run(self, id: int):
        if self.is_first_run:
            self.first_run(id)
            self.is_first_run = False
            return True
        else:
            return False
    
    @abstractmethod
    def first_run(self, id: int):
        pass

class FinalizingStepClass(StepClass):
    _instances = []
    def __init__(self):
        super().__init__()
        FinalizingStepClass._instances.append(self)

    def execute(self):
        for step in FinalizingStepClass._instances:
            print('++++++++++++++++++++++++++++')
            print('Running : ', step)
            print('++++++++++++++++++++++++++++')
            step.run()

class IndependentStepClass(StepClass):
    _instances = []
    def __init__(self):
        super().__init__()
        IndependentStepClass._instances.append(self)

    def execute(self):
        for step in IndependentStepClass._instances:
            print('++++++++++++++++++++++++++++')
            print('Running : ', step)
            print('++++++++++++++++++++++++++++')
            step.run()
