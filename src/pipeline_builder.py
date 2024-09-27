import pickle

from src import Experiment, Settings, ScopeClass, Pycromanager2NativeDataType, DataContainer, Pipeline, \
    ConsolidateImageShapes, TrimZSlices, CalculateSharpness, AutomaticSpotDetection, CellSegmentationStepClass, \
    SpotDetectionStepClass, SavePDFReport


#%% User specified Stuff
name = 'Luis_Standard_Pipeline'

raw_data_location = r"C:\Users\Jack\Desktop\H128_Tiles_100ms_5mW_Blue_15x15_10z_05step_2"

FISHChannel = [0]
NucChannel = [0]

voxel_size_z = 500


#%% Pipeline
# Settings
settings = Settings()

# Scope
scope = ScopeClass()

# Experiment
experiment = Experiment(raw_data_location, voxel_size_z=voxel_size_z)

# Bridge
bridge = Pycromanager2NativeDataType

# PrePipeline Steps
preSteps = [ConsolidateImageShapes(), TrimZSlices(), CalculateSharpness(), AutomaticSpotDetection()]

# Pipeline Steps
Steps = [CellSegmentationStepClass(), SpotDetectionStepClass()]

# PostPipeline Step
postSteps = [SavePDFReport()]


#%% Compile into a dictionary
# Compile
pipelineDict = {
    'settings': settings,
    'experiment': experiment,
    'scope': scope,
    'bridge': bridge,
    'prepipeline_steps': preSteps,
    'pipeline_steps': Steps,
    'postpipeline_steps': postSteps,
    'kwargs': {}
}

# Pickle
with open(f'{name}.pkl', 'wb') as f:
    pickle.dump(pipelineDict, f)


