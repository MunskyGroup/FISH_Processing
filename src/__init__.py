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
from .PipelineSteps import CellSegmentationOutput, CellSegmentationStepClass, SpotDetectionStepOutputClass, SpotDetectionStepClass
from .PrePipelineSteps import ConsolidateImageShapes, CalculateSharpness, AutomaticSpotDetection, TrimZSlices
from .PostPipelineSteps import SavePDFReport, AddTPs2FISHDataFrame
from . import Util
from .DataTypeBridges import Pycromanager2NativeDataType #DataTypeBridges
from .Pipeline import Pipeline


#Importing modules