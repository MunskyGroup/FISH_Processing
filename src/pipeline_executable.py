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
warnings.filterwarnings("ignore")
from src import Pipeline
######################################
######################################

# Zipped pickle files
pipeline_package_location = sys.argv[0]
extract_location = os.path.join(os.getcwd(), 'pipeline_package')

# Unzipping the pickle files
shutil.unpack_archive(pipeline_package_location, extract_dir=extract_location)

# Loading the pickle files
Settings = pickle.load(open(os.path.join(extract_location, 'settings.pkl'), 'rb'))
Scope = pickle.load(os.path.join(extract_location, 'scope.pkl'), 'rb')
Experiment = pickle.load(os.path.join(extract_location, 'experiment.pkl'), 'rb')
Data = pickle.load(os.path.join(extract_location, 'data.pkl'), 'rb')
PrePipelineSteps = pickle.load(os.path.join(extract_location, 'prepipeline_steps.pkl'), 'rb')
PostPipelineSteps = pickle.load(os.path.join(extract_location, 'postpipeline_steps.pkl'), 'rb')
PipelineSteps = pickle.load('pipeline_steps.pkl', 'rb')


# Running the pipeline
pipeline = Pipeline(Settings, Scope, Experiment, Data, PrePipelineSteps, PostPipelineSteps, PipelineSteps)

pipeline.run_pre_pipeline_steps()

pipeline.run_pipeline_steps()

pipeline.run_post_pipeline_steps()
