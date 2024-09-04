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

from .PipelineSteps import (CellSegmentationOutput, CellSegmentationStepClass, SpotDetectionStepOutputClass,
                            SpotDetectionStepClass, BIGFISH_SpotDetection, BIGFISH_Tensorflow_Segmentation, CellSegmentationStepClass_JF)
from .PrePipelineSteps import ConsolidateImageShapes, CalculateSharpness, AutomaticSpotDetection, TrimZSlices, Make_Output_Dir, Make_Output_Dir_JF, \
                                AutomaticSpotDetection_JF, Make_Analysis_Dir_JF
from .PostPipelineSteps import SavePDFReport, Luis_Additional_Plots, Move_Results_To_Analysis_Folder, Send_Data_To_Nas, BuildPDFReport, SaveSpotDetectionResults

from . import Util
from .DataTypeBridges import Pycromanager2NativeDataType #DataTypeBridges
from .Pipeline import Pipeline


#Importing modules