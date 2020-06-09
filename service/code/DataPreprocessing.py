from azureml.core.runconfig import RunConfiguration
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
import json
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace,Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep
import os

with open("./configuration/config.json") as f:
    config = json.load(f)

workspace_name = config["workspace_name"]
resource_group = config["resource_group"]
subscription_id = config["subscription_id"]
location = config["location"]

#cli_auth = AzureCliAuthentication()


# Get workspace
#ws = Workspace.from_config(auth=cli_auth)
ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group
        #auth=cli_auth
    )

# Attach Experiment
experiment_name = "XRayML_AzureDevOps"
exp = Experiment(workspace=ws, name=experiment_name)
print(exp.name, exp.workspace.name, sep="\n")

# Editing a run configuration property on-fly.
run_config_user_managed = RunConfiguration()
run_config_user_managed.environment.python.user_managed_dependencies = True



print("Pipeline SDK-specific imports completed")

#ws = Workspace.from_config()
datastore = Datastore.get(ws,"xray_datastore")

PreProcessingData = PipelineData("PreProcessingData", datastore=datastore)
 
preprocessing_step = PythonScriptStep(name="preprocessing_step",
                                      script_name="data_preprocess/estimator_data_preprocessing.py", 
                                      compute_target=CPU_compute_target, 
                                      runconfig = run_config_user_managed,
                                      source_directory = './scripts',
                                      inputs=[xrayimage_dataset.as_named_input('xrayimage_dataset').as_mount('/temp/xray_images'),
                                              traindata_dataset.as_named_input('traindata_dataset'),
                                              validdata_dataset.as_named_input('validdata_dataset'),
                                              testdata_dataset.as_named_input('testdata_dataset'),
                                              traintarget_dataset.as_named_input('traintarget_dataset'),
                                              validtarget_dataset.as_named_input('validtarget_dataset'),
                                              testtarget_dataset.as_named_input('testtarget_dataset')],
                                      arguments=['--PreProcessingData', PreProcessingData], 
                                      outputs = [PreProcessingData],
                                      allow_reuse=True)

print("preprocessing_step")


run = exp.submit(preprocessing_step)

# Shows output of the run on stdout.
run.wait_for_completion(show_output=True, wait_post_processing=True)

# Raise exception if run fails
if run.get_status() == "Failed":
    raise Exception(
        "Training on local failed with following run status: {} and logs: \n {}".format(
            run.get_status(), run.get_details_with_logs()
        )
    )

# Writing the run id to /aml_config/run_id.json

run_id = {}
run_id["run_id"] = run.id
run_id["experiment_name"] = run.experiment.name
with open("./configuration/run_id.json", "w") as outfile:
    json.dump(run_id, outfile)
    
    