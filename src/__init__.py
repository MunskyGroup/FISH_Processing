# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:16 2020

@author: luisaguilera
"""

name = 'fish_analyses'

#Package version
__version__ = "0.0.1"                                                                   

#Importing modules

from .PipelineSettings import PipelineSettings
from .ExperimentClass import Experiment #ExperimentClass
from .MicroscopeClass import ScopeClass
from .GeneralStepClasses import StepClass, PipelineStepsClass, postPipelineStepsClass, prePipelineStepsClass   #GeneralStepClasses
from .GeneralOutputClasses import OutputClass, StepOutputsClass, PipelineOutputsClass, PrePipelineOutputsClass # GeneralOutputClasses
from .PipelineDataClass import PipelineDataClass
from .Pipeline import Pipeline
from .SingleStepCompiler import SingleStepCompiler

from . import Util
from . import PipelineSteps
from . import PrePipelineSteps
from . import PostPipelineSteps

#Importing modules