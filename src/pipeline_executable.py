
import torch
# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Get the name of each GPU
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

    # Get the current GPU memory usage
    for i in range(num_gpus):
        gpu_memory_allocated = torch.cuda.memory_allocated(i)
        gpu_memory_reserved = torch.cuda.memory_reserved(i)
        print(f"GPU {i} memory allocated: {gpu_memory_allocated / (1024 ** 3):.2f} GB")
        print(f"GPU {i} memory reserved: {gpu_memory_reserved / (1024 ** 3):.2f} GB")
else:
    print("CUDA is not available.")

import sys
import os
import json
import logging
import numba
import matplotlib
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').disabled = True
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

src_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(src_path)

from src.Parameters import Parameters
from src import StepClass
from src.Pipeline import Pipeline

#%% 
pipeline_location = os.path.normpath(sys.argv[1])

with open(pipeline_location, 'r') as f:
    pipeline_dict = json.load(f)


repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
connection_config_location = str(os.path.join(repo_path, 'config_nas.yml'))
for i in range(len(pipeline_dict['params'])):
    pipeline_dict['params'][i]['display_plots'] = False
    pipeline_dict['params'][i]['connection_config_location'] = connection_config_location

    
steps = pipeline_dict['steps']
experiment_locations = [i['initial_data_location'] for i in pipeline_dict['params']]

print(steps)
print(pipeline_dict['params'])

pipeline = Pipeline(experiment_location=experiment_locations, parameters=pipeline_dict['params'], steps=steps)

pipeline.run()
