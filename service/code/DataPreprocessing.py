from azureml.core.runconfig import RunConfiguration
from azureml.core import Experiment,ScriptRunConfig
from azureml.core import Workspace,Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import json,os

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
# Create CPU cluster for Data preprocessing
cpu_cluster_name = "cpu-cluster"

try:
    CPU_compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cpu compute target')
except ComputeTargetException:
    print('Creating a new cpu compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D11', 
                                                           max_nodes=2)

    # create the cluster
    CPU_compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    CPU_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=10)

# use get_status() to get a detailed status for the current cluster. 
print(CPU_compute_target.get_status().serialize())

#######################################################################################################
#Creating dataset reference
xrayimage_dataset = Dataset.get_by_name(ws, name='xray_image_ds')
traindata_dataset = Dataset.get_by_name(ws, name='train_data_ds')
validdata_dataset = Dataset.get_by_name(ws, name='valid_data_ds')
testdata_dataset = Dataset.get_by_name(ws, name='test_data_ds')
traintarget_dataset = Dataset.get_by_name(ws, name='train_target_ds')
validtarget_dataset = Dataset.get_by_name(ws, name='valid_target_ds')
testtarget_dataset = Dataset.get_by_name(ws, name='test_target_ds')

datastore = Datastore.get(ws,"xray_datastore")

PreProcessingData = PipelineData("PreProcessingData", datastore=datastore)
####################################################################################################### 
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
    
    