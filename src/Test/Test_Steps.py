import pytest
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..'))

from src.Parameters import Parameters, ScopeClass, Experiment, DataContainer, Settings

# To Test: These are the most basic steps we will ever use so they must all work always.
from src.GeneralStep import SequentialStepsClass, IndependentStepClass, FinalizingStepClass, StepClass
from src.FinalizationSteps import Save_Images, Save_Masks, Save_Outputs, return_to_NAS, remove_local_data, remove_local_data_but_keep_h5
from src.SequentialSteps import SimpleCellposeSegmentaion, BIGFISH_SpotDetection
from src.IndependentSteps import FFF2NativeDataType, Pycromanager2NativeDataType, NativeDataType

def test_sanity():
    # Setup
    # exercise
    # assert
    assert True == True

























