from azureml.core.runconfig import RunConfiguration
from azureml.core import Experiment,ScriptRunConfig
from azureml.core.authentication import ServicePrincipalAuthentication
import json,os
from azureml.core import Workspace,Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep,EstimatorStep
from azureml.train.dnn import TensorFlow
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Dataset

#######################################################################################################
with open("./configuration/config.json") as f:
    config = json.load(f)

workspace_name = config["workspace_name"]
resource_group = config["resource_group"]
subscription_id = config["subscription_id"]
location = config["location"]

sp_key= config["sp_key"]
sp_app_id= config["sp_app_id"]
sp_tenant_id= config["sp_tenant_id"]

#cli_auth = AzureCliAuthentication()

az_sp = ServicePrincipalAuthentication(sp_tenant_id, sp_app_id, sp_key)

# Get workspace
#ws = Workspace.from_config(auth=cli_auth)
ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=az_sp
    )

# Attach Experiment
experiment_name = "XRayML_AzureDevOps"
exp = Experiment(workspace=ws, name=experiment_name)
print(exp.name, exp.workspace.name, sep="\n")

# Editing a run configuration property on-fly.
run_config_user_managed = RunConfiguration()
run_config_user_managed.environment.python.user_managed_dependencies = True

print("Pipeline SDK-specific imports completed")
#######################################################################################################
# Ceate GPU cluster for training

gpu_cluster_name = "gpu-cluster"

try:
    GPU_compute_target = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print('Found existing GPU compute target')
except ComputeTargetException:
    print('Creating a new GPU compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=4)

    # create the cluster
    GPU_compute_target = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    GPU_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=10)

# use get_status() to get a detailed status for the current cluster. 
print(GPU_compute_target.get_status().serialize())

#######################################################################################################
#Creating dataset reference
datastore = Datastore.get(ws,"xray_datastore")

PreProcessingData = PipelineData("PreProcessingData", datastore=datastore)
#######################################################################################################

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


#######################################################################################################
pipeline = Pipeline(workspace = ws,steps=[est_step])

#Validate pipeline
pipeline.validate()
print("Pipeline validation complete")

#submit Pipeline
run = exp.submit(pipeline,pipeline_parameters={})
print("Pipeline is submitted for execution")

#######################################################################################################
# Shows output of the run on stdout.
run.wait_for_completion(show_output=True)

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
    
    