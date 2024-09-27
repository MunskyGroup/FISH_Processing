import pickle
import sys

from src import DataContainer, Pipeline

# load in args
pipeline_dict = pickle.load(open(sys.argv[1], "rb"))
config_location = sys.argv[2]

# Separate into individual objects
settings = pipeline_dict['settings']
experiment = pipeline_dict['experiment']
scope = pipeline_dict['scope']
preSteps = pipeline_dict['prepipeline_steps']
postSteps = pipeline_dict['postpipeline_steps']
Steps = pipeline_dict['pipeline_steps']
additional_settings = pipeline_dict['kwargs']  # TODO: This could be used to change the behaviour in the future

bridge = pipeline_dict.get('bridge')
if bridge is not None:
    native_data = bridge(experiment, settings, scope, config_location)
    Data = DataContainer(native_data.local_data_dir, total_num_imgs=experiment.number_of_images_to_process,
                             list_image_names=native_data.list_files_names, list_images=native_data.list_images)
else:
    raise Exception('No bridge found, no way to download the data and convert to [[z, x, y, c], ...] format')
    # TODO: I want this to change behaviour in the future.

# run pipeline
pipeline = Pipeline(settings, scope, experiment, Data, preSteps, postSteps, Steps)

pipeline.execute_independent_steps()

pipeline.execute_finalization_steps()

pipeline.execute_finalization_steps()



