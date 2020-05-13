#
# @license
# Copyright 2020 Cloudera Fast Forward. https://github.com/fastforwardlabs
# DeepAd: Experiments for Deep Learning for Anomaly Detection https://ff12.fastforwardlabs.com/.
# Licensed under the MIT License (the "License");
# =============================================================================
#

import os
import time
import datetime
from cmlbootstrap import CMLBootstrap

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
                      "script": "cml/train_model.py",
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
