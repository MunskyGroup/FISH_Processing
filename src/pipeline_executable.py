#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 00:00:00 2022

@author: luis_aguilera
"""


######################################
######################################
# Importing libraries
import sys
import pathlib
import warnings
import json
import pickle
import shutil 
import os
import numpy as np
warnings.filterwarnings("ignore")
from src import Pipeline, Settings, ScopeClass, Experiment, DataContainer, \
    IndependentSteps, FinalizationSteps, SequentialSteps
######################################
######################################
def load_dict_from_file(location):
    f = open(location,'r')
    data=f.read()
    f.close()
    return eval(data)

# Zipped pickle files
pipeline_package_location = os.path.normpath(sys.argv[1])
# extract_location = os.path.join(os.getcwd(), 'pipeline_package', os.path.splitext(os.path.basename(pipeline_package_location))[0])

# Unzipping the pickle files
# shutil.unpack_archive(pipeline_package_location, extract_dir=extract_location, format='zip')
# extract_location = os.path.join(extract_location, os.path.splitext(os.path.basename(pipeline_package_location))[0])

# Loading the pickle files
pipeline_dict = load_dict_from_file(pipeline_package_location)

settings = pipeline_dict['settings']
scope = pipeline_dict['scope']
experiment = pipeline_dict['experiment']
Settings = Settings(**settings)
Scope = ScopeClass(**scope)
experiment = Experiment(**experiment)
Data = DataContainer(**pipeline_dict['data'])

prePipelineSteps = [getattr(IndependentSteps, i)() for i in pipeline_dict['PrePipelineSteps']]
postPipelineSteps = [getattr(FinalizationSteps, i)() for i in pipeline_dict['PostPipelineSteps']]
pipelineSteps = [getattr(SequentialSteps, i)() for i in pipeline_dict['PipelineSteps']]

# Running the pipeline
pipeline = Pipeline(Settings, Scope, experiment, Data, prePipelineSteps, postPipelineSteps, pipelineSteps)

pipeline.execute_independent_steps()

pipeline.execute_sequential_steps()

pipeline.execute_finalization_steps()
