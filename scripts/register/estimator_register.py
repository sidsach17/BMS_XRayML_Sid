import os
import sys
import argparse

from azureml.core import Run
from azureml.core.model import Model
from azureml.pipeline.steps import HyperDriveStep, HyperDriveStepRun
######################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--ModelData', dest='ModelData', required=True)
args = parser.parse_args()

Model_OutputPath=args.ModelData

######################################################################################################

run= Run.get_context()
run_id = run.parent.id

print("register final model")

parent_run = Run(experiment = run.experiment,run_id=run_id)


model = parent_run.register_model(model_name='MLOps_Model', model_path='outputs/weights.best.dense_generator_callback.hdf5')

######################################################################################################

os.makedirs(Model_OutputPath, exist_ok=True)

model_json = {}
model_json["model_name"] = model.name
model_json["model_version"] = model.version
with open(os.path.join(Model_OutputPath,'model.json'), "w") as outfile: 
    json.dump(model_json, outfile) 