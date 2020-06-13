#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments for Deep Learning for Anomaly Detection https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

!pip3  install -r requirements.txt

import os
import time
import datetime
from cmlbootstrap import CMLBootstrap
import numpy as np
run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")


HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split("/")[6]
API_KEY = os.getenv("CDSW_API_KEY")
PROJECT_NAME = os.getenv("CDSW_PROJECT")


cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)


# Get User Details
user_details = cml.get_user({})
user_obj = {"id": user_details["id"], "username": "vdibia",
            "name": user_details["name"],
            "type": user_details["type"],
            "html_url": user_details["html_url"],
            "url": user_details["url"]
            }


# Get Project Details
project_details = cml.get_project({})
project_id = project_details["id"]


# Create Job
create_jobs_params = {"name": "Train Autoencoder Model " + run_time_suffix,
                      "type": "manual",
                      "script": "train.py",
                      "timezone": "America/Los_Angeles",
                      "environment": {},
                      "kernel": "python3",
                      "cpu": 1,
                      "memory": 2,
                      "nvidia_gpu": 0,
                      "include_logs": True,
                      "notifications": [
                          {"user_id": user_obj["id"],
                           "user":  user_obj,
                           "success": False, "failure": False, "timeout": False, "stopped": False
                           }
                      ],
                      "recipients": {},
                      "attachments": [],
                      "include_logs": True,
                      "report_attachments": [],
                      "success_recipients": [],
                      "failure_recipients": [],
                      "timeout_recipients": [],
                      "stopped_recipients": []
                      }

new_job = cml.create_job(create_jobs_params)
new_job_id = new_job["id"]
print("Created new job with jobid", new_job_id)

##
# Start a job
job_env_params = {}
start_job_params = {"environment": job_env_params}
job_status = cml.start_job(new_job_id, start_job_params)
print("Job started")

# Create model build script
cdsw_script = """#!/bin/bash
pip3 install -r requirements.txt"""

with open("cdsw-build.sh", 'w+') as f:
    f.write(cdsw_script)
    f.close()
os.chmod("cdsw-build.sh", 0o777)

# Get Default Engine Details
# Engine id is required for next step (create model)

default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]

# Create Model
example_model_input = np.random.rand(3, 18).tolist()
example_model_output = {'scores': [
    0.3811887163134269, 0.34900869152707426, 0.25317983491992346], 'predictions': [True, True, True]}
create_model_params = {
    "projectId": project_id,
    "name": "Anomaly Detection " + run_time_suffix,
    "description": "Predict if data is normal or abnormal",
    "visibility": "private",
    "targetFilePath": "cml_servemodel.py",
    "targetFunctionName": "predict",
    "engineImageId": default_engine_image_id,
    "kernel": "python3",
    "examples": [
        {
            "request": example_model_input,
            "response": example_model_output
        }],
    "cpuMillicores": 1000,
    "memoryMb": 2048,
    "nvidiaGPUs": 0,
    "replicationPolicy": {"type": "fixed", "numReplicas": 1},
    "environment": {}}

new_model_details = cml.create_model(create_model_params)
access_key = new_model_details["accessKey"]  # todo check for bad response
print("New model created with access key", access_key)


# Wait for the model to deploy.
is_deployed = False
while is_deployed == False:
    model = cml.get_model({"id": str(
        new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
    if model["latestModelDeployment"]["status"] == 'deployed':
        print("Model is deployed")
        break
    else:
        print("Model deployment status .....",
              model["latestModelDeployment"]["status"])
        time.sleep(10)
