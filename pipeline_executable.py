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
from src import Pipeline, PipelineSettings, ScopeClass, Experiment
######################################
######################################
def load_dict_from_file(location):
    f = open(location,'r')
    data=f.read()
    f.close()
    return eval(data)

# Zipped pickle files
pipeline_package_location = os.path.normpath(sys.argv[1])
extract_location = os.path.join(os.getcwd(), 'pipeline_package')

# Unzipping the pickle files
shutil.unpack_archive(pipeline_package_location, extract_dir=extract_location, format='zip')
extract_location = os.path.join(extract_location, os.path.splitext(os.path.basename(pipeline_package_location))[0])

# Loading the pickle files
settings = load_dict_from_file(os.path.join(extract_location, 'settings.txt'))
scope = load_dict_from_file(os.path.join(extract_location, 'scope.txt'))
experiment = load_dict_from_file(os.path.join(extract_location, 'experiment.txt'))
# settings = np.load(os.path.join(extract_location, 'settings.npy')).item()
# scope = np.load(os.path.join(extract_location, 'scope.npy')).item()
# experiment = np.load(os.path.join(extract_location, 'experiment.npy')).item()
# Settings = pickle.load(open(os.path.join(extract_location, 'settings.pkl'), 'rb'))
# Scope = pickle.load(open(os.path.join(extract_location, 'scope.pkl'), 'rb'))
# Experiment = pickle.load(open(os.path.join(extract_location, 'experiment.pkl'), 'rb'))
Data = pickle.load(open(os.path.join(extract_location, 'data.pkl'), 'rb'))
PrePipelineSteps = pickle.load(open(os.path.join(extract_location, 'prepipeline_steps.pkl'), 'rb'))
PostPipelineSteps = pickle.load(open(os.path.join(extract_location, 'postpipeline_steps.pkl'), 'rb'))
PipelineSteps = pickle.load(open(os.path.join(extract_location, 'pipeline_steps.pkl'), 'rb'))

Settings = PipelineSettings(**settings)
Scope = ScopeClass(**scope)
Experiment = Experiment(**experiment)

# Running the pipeline
pipeline = Pipeline(Settings, Scope, Experiment, Data, PrePipelineSteps, PostPipelineSteps, PipelineSteps)

pipeline.run_pre_pipeline_steps()

pipeline.run_pipeline_steps()

pipeline.run_post_pipeline_steps()
