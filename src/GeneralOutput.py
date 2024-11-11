from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod
# Many of the output classes will be the same, so we can create a base class and then inherit from it
# however, they will have differences on if they modify a PipelineDataClass or if they are the final output

# The Step Classes will be similar however they will have differences on the inputs they act on 
class OutputClass(ABC):
    """ This class will be used to generate singletons for the output classes. """
    _instances = []
    def __init__(self):
        OutputClass._instances.append(self)

    @classmethod
    def get_all_instances(cls):
        # Class method to return all instances of the parent class
        return cls._instances
    
    def __new__(cls, *args, **kwargs):
        for instance in cls._instances:
            if isinstance(instance, cls):
                return instance
        instance = super().__new__(cls)
        cls._instances.append(instance)
        return instance
    
    @abstractmethod
    def __init__(self, value=None):
        if not hasattr(self, '_initialized'):
            self._initialized = True  # Mark the instance as initialized
            self.value = value
        else:
            self.append()

    @abstractmethod
    def append(self):
        pass


class StepOutputsClass(OutputClass):
    # this will be the final output of the pipeline, this will be the final output of the pipeline
    def __init__(self):
        super().__init__()

    def append(self, newOutputs):
        pass


class PipelineOutputsClass(OutputClass):
    # in pipelineData we will store many of these, these will consist of the direct results from the steps.
    # This could const of dataframes, images, etc.
    # the append function will be used to add the output to add the outputs from each image to itself.
    def __init__(self):
        super().__init__()

    def append(self, output:StepOutputsClass):
        if hasattr(self, output.__class__.__name__):
            getattr(self, output.__class__.__name__).append(output)
        else:
            setattr(self, output.__class__.__name__, output)


class PrePipelineOutputsClass(OutputClass):
    # this will be the final output of the pipeline, this will be the final output of the pipeline
    def __init__(self):
        super().__init__()

    def append(self, newOutputs):
        setattr(self, newOutputs.__class__.__name__, newOutputs)