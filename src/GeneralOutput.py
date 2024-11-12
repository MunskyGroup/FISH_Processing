from typing import List, Dict, Any, Union
from abc import ABC, abstractmethod
# Many of the output classes will be the same, so we can create a base class and then inherit from it
# however, they will have differences on if they modify a PipelineDataClass or if they are the final output

# The Step Classes will be similar however they will have differences on the inputs they act on 
class OutputClass(ABC):
    """ This class will be used to generate singletons for the output classes. """
    _instances = []

    def __new__(cls, *args, **kwargs):
        for instance in OutputClass._instances:
            if isinstance(instance, cls):
                instance.append(*args, **kwargs)
                return instance
        instance = super().__new__(cls)
        OutputClass._instances.append(instance)
        return instance
    
    def __init__(self, *args, **kwargs):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.append(*args, **kwargs)

    @classmethod
    def get_all_instances(cls):
        # Class method to return all instances of the parent class
        return cls._instances

    @abstractmethod
    def append(self, *args, **kwargs):
        pass

    @classmethod
    def clear_instances(cls):
        # del all instances out of memory
        for instance in cls._instances:
            del instance

