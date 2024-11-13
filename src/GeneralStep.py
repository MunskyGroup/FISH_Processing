import os
from abc import ABC, abstractmethod
import inspect

from .Parameters import Parameters
# from . import Settings, Experiment, ScopeClass, DataContainer


class StepClass(ABC):
    _instances = []

    def __new__(cls, *args, **kwargs):
        for instance in cls._instances:
            if isinstance(instance, cls):
                return instance
        instance = super().__new__(cls)
        cls._instances.append(instance)
        return instance

    @classmethod
    def get_all_instances(cls):
        # Class method to return all instances of the parent class
        return cls._instance

    @classmethod
    def get_step_parameters(cls):
        # inspects each steps main function to see if it has the required parameters
        steps = cls.get_all_instances()
        no_default_params = []
        all_params = []
        for step in steps:
            required, all, _, _ = step.get_paramaters()
            no_default_params.extend(required)
            all_params.extend(all)

        return list(set(no_default_params)), list(set(all_params))

    def __str__(self):
        return self.__class__.__name__

    def load_in_parameters(self, p: int = None, t: int = None):
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
            params['image'] = params['images'][p, t, :, :, :, :]
            try:
                params['cell_mask'] = params['masks'][p, t, params['cytoChannel'], :, :, :] if params['masks'].shape[1] > 1 else params['masks'][p, 0, params['cytoChannel'], :, :, :]
            except AttributeError:
                params['cell_mask'] = None
            try:
                params['nuc_mask'] = params["masks"][p, t, params['nucChannel'], :, :, :] if params['masks'].shape[1] > 1 else params['masks'][p, 0, params['nucChannel'], :, :, :]
            except AttributeError:
                params['nuc_mask'] = None

            if params['cell_mask'] is not None and params['nuc_mask'] is not None:
                params['cyto_mask'] = params['cell_mask']
                params['cyto_mask'][params['nuc_mask'] >= 1] = 0
            else:
                params['cyto_mask'] = None
            
        
        return params
    
    def create_step_output_dir(self, output_location = None, **kwargs):
        if output_location is not None:
            self.step_output_dir = os.path.join(output_location, self.__class__.__name__)
            os.makedirs(self.step_output_dir, exist_ok=True)
        else:
            self.step_output_dir = None

    def get_parameters(self):
            step_func = self.main
            sig = inspect.signature(step_func)
            # get the required parameters
            required_params = [param.name for param in sig.parameters.values() if param.default is param.empty]
            all_params = [param.name for param in sig.parameters.values()]
            
            # get default values for all parameters
            defaults = {param.name: param.default for param in sig.parameters.values() if param.default is not param.empty}

            # get types for all parameters
            types = {param.name: (param.annotation if param.annotation is not param.empty else None) for param in sig.parameters.values()}

            return required_params, all_params, defaults, types
    
    @abstractmethod
    def main(self, **kwargs):
        pass

    def run(self, p: int = None, t:int = None):
        kwargs = self.load_in_parameters(p, t)
        return self.main(**kwargs) 

class SequentialStepsClass(StepClass):
    order = 'pt'
    _instances = []

    def __init__(self):
        StepClass._instances.append(self)
        SequentialStepsClass._instances.append(self)
        self.is_first_run = True

    @classmethod
    def execute(self):
        params = Parameters.get_parameters()
        number_of_chunks = params['num_chunks_to_run']
        count = 0
        if SequentialStepsClass.order == 'tp':
            for t in range(params['images'].shape[1]):
                if count >= number_of_chunks:
                    break
                for p in range(params['images'].shape[0]):
                    if count >= number_of_chunks:
                        break
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

        elif SequentialStepsClass.order == 'pt':
            for p in range(params['images'].shape[0]):
                if count >= number_of_chunks:
                    break
                for t in range(params['images'].shape[1]):
                    if count >= number_of_chunks:
                        break
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
        else:
            raise ValueError('Order must be either "pt" or "tp"')

    def run(self, p:int = None, t:int = None):
        if p is None and t is None:
            params = Parameters.get_parameters()
            number_of_chunks = params['num_chunks_to_run']
            count = 0
            if SequentialStepsClass.order == 'tp':
                print('++++++++++++++++++++++++++++')
                print('Running : ', self)
                print('++++++++++++++++++++++++++++')
                print('')
                for t in range(params['images'].shape[1]):
                    if count >= number_of_chunks:
                        break
                    for p in range(params['images'].shape[0]):
                        if count >= number_of_chunks:
                            break
                        print(' ###################### ')
                        print('FOV' + str(p) + ' TIMEPOINT : ' + str(t))
                        print(' ###################### ')
                        params = self.load_in_parameters(p, t)
                        self.create_step_output_dir(**params)
                        self.on_first_run()
                        output = self.main(**params)
                        count += 1
            elif SequentialStepsClass.order == 'pt':
                print('++++++++++++++++++++++++++++')
                print('Running : ', self)
                print('++++++++++++++++++++++++++++')
                print('')
                for p in range(params['images'].shape[0]):
                    if count >= number_of_chunks:
                        break
                    for t in range(params['images'].shape[1]):
                        if count >= number_of_chunks:
                            break
                        print(' ###################### ')
                        print('FOV:' +  str(p) + ' TIMEPOINT: ' + str(t))
                        print(' ###################### ')
                        params = self.load_in_parameters(p, t)
                        self.create_step_output_dir(**params)
                        self.on_first_run(params)
                        output = self.main(**params)
                        count += 1
                        if count >= number_of_chunks:
                            break
        elif p is not None and t is not None:
            print('')
            print(' ###################### ')
            print('FOV:' + str(p) + ' TIMEPOINT : ' + str(t))
            print(' ###################### ')
            params = self.load_in_parameters(p, t)
            self.create_step_output_dir(**params)
            self.on_first_run(params)
            output = self.main(**params)
        
        return output

    def main(self, **kwargs):
        pass

    def on_first_run(self, params):
        if self.is_first_run:
            self.first_run(params)
            self.is_first_run = False
            return True
        else:
            return False
    
    def first_run(self, params):
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
