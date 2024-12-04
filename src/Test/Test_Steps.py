import pytest
import sys
import os
from unittest.mock import patch, MagicMock


sys.path.append(os.path.join(os.getcwd(), '..'))

from src.Parameters import Parameters, ScopeClass, Experiment, DataContainer, Settings

# To Test: These are the most basic steps we will ever use so they must all work always.
from src.GeneralStep import SequentialStepsClass, IndependentStepClass, FinalizingStepClass, StepClass
from src.FinalizationSteps import Save_Images, Save_Masks, Save_Outputs, return_to_NAS, remove_local_data, remove_local_data_but_keep_h5
from src.SequentialSteps import SimpleCellposeSegmentaion, BIGFISH_SpotDetection
from src.IndependentSteps import NativeDataType

def test_sanity():
    # Setup
    # exercise
    # assert
    assert True == True

def setUp():
    dataBase_loc = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dataBase_loc = os.path.join(dataBase_loc, 'dataBase')
    
    example_data = os.path.join(dataBase_loc, 'example_data')

    scope = ScopeClass() 
    data = DataContainer(local_dataset_location=example_data) 
    settings = Settings(name='Test') 
    experiment = Experiment()

    experiment.FISHChannel = 0
    experiment.nucChannel = 2
    experiment.cytoChannel = 1
    experiment.voxel_size_z = 500

    settings.num_chunks_to_run = 1

    scope.spot_yx = 130
    scope.spot_z = 360
    scope.voxel_size_yx = 100


def test_parameter_initiation():
    pass

def test_step_initiation():
    pass

def test_load_in_data():
    pass

def test_load_in_mask():
    pass

def test_segementation():
    pass

def test_spot_detection():
    pass

def test_save_images():
    pass

def test_save_params():
    pass
    
def test_save_output():
    pass


















