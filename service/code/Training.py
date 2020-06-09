from azureml.core.runconfig import RunConfiguration
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
import json
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace,Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep
import os
from azureml.pipeline.steps import EstimatorStep
from azureml.train.dnn import TensorFlow

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

est = TensorFlow(source_directory = './scripts', 
                    compute_target = GPU_compute_target,
                    entry_script = "train/estimator_training.py",
                    pip_packages = ['keras<=2.3.1','matplotlib','opencv-python','azure-storage-blob==2.1.0','tensorflow-gpu==2.0.0'],
                    conda_packages = ['scikit-learn==0.22.1'],
                    use_gpu = True )


est_step = EstimatorStep(name="Estimator_Train", 
                         estimator=est, 
                         estimator_entry_script_arguments=['--PreProcessingData', PreProcessingData],
                         inputs=[PreProcessingData],
                         runconfig_pipeline_params=None,
                         compute_target=compute_target)


run = exp.submit(est_step)

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
    
    