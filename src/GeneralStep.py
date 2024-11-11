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
            _1, _2 = step.get_paramaters()
            no_default_params.extend(_1)
            all_params.extend(_2)

        return list(set(no_default_params)), list(set(all_params))
    
    def check_setting_requirements(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def load_in_attributes(self, t: int = None, p: int = None):
        """
        This is where the magic happens. This function will load in all the attributes of the class and return them as a dictionary.
        This allows all the bs that I decided to force on my code to not matter, and user can just write whatever they want in the main functions
        As long as the attributes are unique and saved using a step output class, this function will load them in.
        """

        params = Parameters.get_parameters()
        if t is None != p is None:
            raise ValueError('t and p must be both None or both not None')

        if t is not None and p is not None: # TODO This will need to be changed for Dask arrays
            params['fov'] = p
            params['timepoint'] = t
            params['image'] = self.data.images[p, t, :, :, :, :]
            try:
                params['cell_mask'] = self.params.masks[p, t, params['cytoChannel'], :, :, :] if self.params.masks.shape[1] > 1 else self.params.masks[p, 0, params['cytoChannel'], :, :, :]
            except AttributeError:
                params['cell_mask'] = None
            try:
                params['nuc_mask'] = self.params.masks[p, t, params['nucChannel'], :, :, :] if self.params.masks.shape[1] > 1 else self.params.masks[p, 0, params['nucChannel'], :, :, :]
            except AttributeError:
                params['nuc_mask'] = None

            if params['cell_mask'] is not None and params['nuc_mask'] is not None:
                params['cyto_mask'] = params['cell_mask']
                params['cyto_mask'][params['nuc_mask'] >= 1] = 0
        
        return params
    
    def create_step_output_dir(self, output_location = None, **kwargs):
        if output_location is not None:
            self.step_output_dir = os.path.join(output_location, self.__class__.__name__)
            os.makedirs(self.step_output_dir, exist_ok=True)
        else:
            self.step_output_dir = None

    def get_paramaters(self):
            step_func = self.main
            sig = inspect.signature(step_func)
            no_default_params = [param.name for param in sig.parameters.values() if param.default is param.empty]
            all_params = [param.name for param in sig.parameters.values()]
            return no_default_params, all_params
    
    @abstractmethod
    def main(self, **kwargs):
        pass

    def run(self, p: int = None, t:int = None):
        kwargs = self.load_in_attributes(p, t)
        return self.main(**kwargs) 

class SequentialStepsClass(StepClass):
    order = 'pt'
    _instances = []

    def __init__(self):
        super().__init__()
        SequentialStepsClass._instances.append(self)
        self.is_first_run = True

    def execute(self):
        params = Parameters.get_parameters()
        number_of_chunks = params['num_chunks_to_run']
        count = 0
        if SequentialStepsClass.order == 'tp':
            for t in range(params['images'].shape[1]):
                for p in range(params['images'].shape[0]):
                    print(' ###################### ')
                    print('        IMAGE : ' + str(p) + ' TIMEPOINT : ' + str(t))
                    print(' ###################### ')
                    print('')
                    for step in SequentialStepsClass._instances:
                        print('++++++++++++++++++++++++++++')
                        print('Running : ', step)
                        print('++++++++++++++++++++++++++++')
                        step.run(p, t)
                    count += 1
                    if count >= number_of_chunks:
                        break
        elif SequentialStepsClass.order == 'pt':
            for p in range(params['images'].shape[0]):
                for t in range(params['images'].shape[1]):
                    print(' ###################### ')
                    print('        IMAGE : ' + str(p) + ' TIMEPOINT : ' + str(t))
                    print(' ###################### ')
                    print('')
                    for step in SequentialStepsClass._instances:
                        print('++++++++++++++++++++++++++++')
                        print('Running : ', step)
                        print('++++++++++++++++++++++++++++')
                        step.run(p, t)
                    count += 1
                    if count >= number_of_chunks:
                        break
        else:
            raise ValueError('Order must be either "pt" or "tp"')

    def run(self, p:int = None, t:int = None):
        if p is None and t is None:
            number_of_chunks = Parameters.get_parameters()['num_chunks_to_run']
            count = 0
            if SequentialStepsClass.order == 'tp':
                print('++++++++++++++++++++++++++++')
                print('Running : ', self)
                print('++++++++++++++++++++++++++++')
                print('')
                for t in range(params['images'].shape[1]):
                    for p in range(params['images'].shape[0]):
                        print(' ###################### ')
                        print('        IMAGE : ' + str(p) + ' TIMEPOINT : ' + str(t))
                        print(' ###################### ')
                        params = self.load_in_attributes(p, t)
                        self.create_step_output_dir(**params)
                        self.on_first_run()
                        output = self.main(**params)
                        count += 1
                        if count >= number_of_chunks:
                            break
            elif SequentialStepsClass.order == 'pt':
                print('++++++++++++++++++++++++++++')
                print('Running : ', self)
                print('++++++++++++++++++++++++++++')
                print('')
                for p in range(params['images'].shape[0]):
                    for t in range(params['images'].shape[1]):
                        print(' ###################### ')
                        print('        IMAGE : ' + str(p) + ' TIMEPOINT : ' + str(t))
                        print(' ###################### ')
                        params = self.load_in_attributes(p, t)
                        self.create_step_output_dir(**params)
                        self.on_first_run()
                        output = self.main(**params)
                        count += 1
                        if count >= number_of_chunks:
                            break
        elif p is not None and t is not None:
            print('')
            print(' ###################### ')
            print('        IMAGE : ' + str(p) + ' TIMEPOINT : ' + str(t))
            print(' ###################### ')
            params = self.load_in_attributes(p, t)
            self.create_step_output_dir(**params)
            self.on_first_run()
            output = self.main(**params)
        
        return output

    def main(self, **kwargs):
        pass

    def on_first_run(self):
        if self.is_first_run:
            self.first_run()
            self.is_first_run = False
            return True
        else:
            return False
    
    @abstractmethod
    def first_run(self):
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
