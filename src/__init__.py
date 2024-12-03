# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:02:16 2020

@author: luisaguilera
"""

name = 'fish_analyses'

#Package version
__version__ = "0.0.1"                                                                   

#Importing modules

from .Parameters import Settings, Experiment, ScopeClass, DataContainer
from .GeneralStep import StepClass, SequentialStepsClass, FinalizingStepClass, IndependentStepClass   #GeneralStepClasses
from .GeneralOutput import OutputClass # GeneralOutputClasses
from .Pipeline import Pipeline
from .Send_To_Cluster import run_on_cluster
from .Displays import Display
from .GUI import GUI, StepGUI

from . import Util
from . import SequentialSteps
from . import IndependentSteps
from . import FinalizationSteps

#Importing modules