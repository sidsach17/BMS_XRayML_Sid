from azureml.core.runconfig import RunConfiguration
from azureml.core import Experiment,ScriptRunConfig
import json,os
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace,Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.steps import EstimatorStep
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

register_step = PythonScriptStep(name = "register_step",
                    script_name= "register/estimator_register.py",
                    runconfig = run_config_user_managed,
                    source_directory = './scripts',
                    compute_target=CPU_compute_target 
                    )


#######################################################################################################
pipeline = Pipeline(workspace = ws,steps=[register_step])

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

model_json = {}
model_json["model_name"] = model.name
model_json["model_version"] = model.version
model_json["run_id"] = run_id
with open("./configuration/model.json", "w") as outfile:
    json.dump(model_json, outfile)
    
    