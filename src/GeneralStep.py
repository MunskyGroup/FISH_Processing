import os
from abc import ABC, abstractmethod
import inspect
from copy import copy

from .Parameters import Parameters, DataContainer
# from . import Settings, Experiment, ScopeClass, DataContainer


class StepClass(ABC):
    _instances = []

    def __new__(cls, *args, **kwargs):
        for instance in cls._instances:
            if isinstance(instance, cls):
                return instance
        instance = super().__new__(cls)
        cls._instances.append(instance)
        StepClass._instances.append(instance)
        return instance

    @classmethod
    def get_all_instances(cls):
        # Class method to return all instances of the parent class
        return cls._instances

    @classmethod
    def get_step_parameters(cls):
        # inspects each steps main function to see if it has the required parameters
        steps = cls.get_all_instances()
        no_default_params = []
        all_params = []
        for step in steps:
            required, all, _, _ = step.get_parameters()
            no_default_params.extend(required)
            all_params.extend(all)

        return list(set(no_default_params)), list(set(all_params))

    @classmethod
    def clear_instances(cls):
        for step in cls._instances:
            del step

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
                cytoChannel = params['cytoChannel']
                if cytoChannel is not None:
                    params['cell_mask'] = params['masks'][p, t, cytoChannel, :, :, :] if params['masks'].shape[1] > 1 else params['masks'][p, 0, cytoChannel, :, :, :]
                else:
                    params['cell_mask'] = None
            except AttributeError:
                params['cell_mask'] = None
            try:
                nucChannel = params['nucChannel']
                if nucChannel is not None:
                    params['nuc_mask'] = params["masks"][p, t, nucChannel, :, :, :] if params['masks'].shape[1] > 1 else params['masks'][p, 0, nucChannel, :, :, :]
                else:
                    params['nuc_mask'] = None
            except AttributeError:
                params['nuc_mask'] = None

            if params['cell_mask'] is not None and params['nuc_mask'] is not None:
                params['cyto_mask'] = copy(params['cell_mask'])
                params['cyto_mask'][params['nuc_mask'] >= 1] = 0
            else:
                params['cyto_mask'] = None

        return params

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
    
    @classmethod
    def initalize_steps_from_list(cls, steps: list):
        # get names of all children of this class even if they are not imported
        seq_children = SequentialStepsClass.list_all_children()
        ind_children = IndependentStepClass.list_all_children()
        fin_children = FinalizingStepClass.list_all_children()

        seq_children_names = [cls.__name__ for cls in seq_children]
        ind_children_names = [cls.__name__ for cls in ind_children]
        fin_children_names = [cls.__name__ for cls in fin_children]

        for step in steps:
            if step in seq_children_names:
                i = seq_children_names.index(step)
                seq_children[i]()
            elif step in ind_children_names:
                i = ind_children_names.index(step)
                ind_children[i]()
            elif step in fin_children_names:
                i = fin_children_names.index(step)
                fin_children[i]()
            else:
                raise ValueError(f'{step} is not a valid step class')

    @classmethod
    def list_all_children(cls):
        children = []
        # check if the class is a subclass of this class
        children.append(cls)

        if hasattr(cls, '__subclasses__') and 'ABCMeta' not in cls.__subclasses__() :
            for c in cls.__subclasses__():
                children.extend(c.list_all_children())


        # return the list of all children and the class so it can be initialized
        return children

    @abstractmethod
    def main(self, **kwargs):
        pass

    def run(self, p: int = None, t:int = None):
        DataContainer().load_temp()
        kwargs = self.load_in_parameters(p, t)
        results = self.main(**kwargs) 
        DataContainer().save_temp()
        return 

class SequentialStepsClass(StepClass):
    order = 'pt'
    _instances = []

    def __init__(self):
        self.is_first_run = True

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
                    for step in SequentialStepsClass._instances:
                        print('++++++++++++++++++++++++++++')
                        print('Running : ', step)
                        print('++++++++++++++++++++++++++++')
                        step.run(p, t)
                    count += 1
        else:
            raise ValueError('Order must be either "pt" or "tp"')

    def run(self, p:int = None, t:int = None):
        DataContainer().load_temp()
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
            output = self.main(**params)
        DataContainer().save_temp()
        return output

    def main(self, **kwargs):
        pass

class FinalizingStepClass(StepClass):
    _instances = []

    def execute(self):
        for step in FinalizingStepClass._instances:
            print('++++++++++++++++++++++++++++')
            print('Running : ', step)
            print('++++++++++++++++++++++++++++')
            step.run()

class IndependentStepClass(StepClass):
    _instances = []

    def execute(self):
        for step in IndependentStepClass._instances:
            print('++++++++++++++++++++++++++++')
            print('Running : ', step)
            print('++++++++++++++++++++++++++++')
            step.run()
