# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:16 2020

@author: luisaguilera
"""

name = 'fish_analyses'

#Package version
__version__ = "0.0.1"                                                                   

#Importing modules

from .PipelineSettings import Settings
from .Experiment import Experiment #ExperimentClass
from .Microscope import ScopeClass
from .GeneralStep import StepClass, SequentialStepsClass, finalizingStepClass, IndependentStepClass   #GeneralStepClasses
from .GeneralOutput import OutputClass, StepOutputsClass, PipelineOutputsClass, PrePipelineOutputsClass # GeneralOutputClasses
from .DataContainer import DataContainer
from .Pipeline import Pipeline
from .SingleStepCompiler import SingleStepCompiler
from .Send_To_Cluster import run_on_cluster

from . import Util
from . import SequentialSteps
from . import IndependentSteps
from . import FinalizationSteps

#Importing modules