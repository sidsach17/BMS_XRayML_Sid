from azureml.core.runconfig import RunConfiguration
from azureml.core import Experiment,ScriptRunConfig
from azureml.core import Workspace,Datastore,Dataset
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep,EstimatorStep
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import json,os
from azureml.train.dnn import TensorFlow
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core import Webservice
from azureml.exceptions import WebserviceException
from azureml.core.environment import Environment
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.webservice import Webservice, AksWebservice

###################################################################################################
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
#########################################################################################################
with open("./configuration/model.json") as modelinput:
    model_json = json.load(modelinput)

model_name=model_json["model_name"]
model_version=model_json["model_version"]

model = Model(workspace=ws, name=model_name, version=model_version)

print("model name: ",model_name," model version: ",model_version)

#########################################################################################################
# Choose a name for your GPU cluster
gpu_cluster_name = "aks-gpu-cluster"

# Verify that cluster does not exist already
try:
    aks_gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print("Found existing gpu cluster")
except ComputeTargetException:
    print("Creating new gpu-cluster")
    
    # Specify the configuration for the new cluster
    compute_config = AksCompute.provisioning_configuration(cluster_purpose=AksCompute.ClusterPurpose.DEV_TEST,
                                                           agent_count=1,
                                                           vm_size="Standard_NC6")
    # Create the cluster with the specified name and configuration
    aks_gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)

    # Wait for the cluster to complete, show the output log
    aks_gpu_cluster.wait_for_completion(show_output=True)

#########################################################################################################
deployment_env = Environment.from_conda_specification(name="deployment_env", file_path="./configuration/deployment_env.yml")
inference_config = InferenceConfig(entry_script="./scripts/score/score.py", environment=deployment_env)


# Set the web service configuration (using default here)
aks_config = AksWebservice.deploy_configuration(cpu_cores=2,
                                               auth_enabled=True, # this flag generates API keys to secure access
                                               memory_gb=8,
                                               #tags={'name': 'mnist', 'framework': 'Keras'},
                                               description='mlxrayaks-latest',
                                               max_request_wait_time=300000,scoring_timeout_ms=300000)

 
#########################################################################################################

service_name = 'mlops-estimator-model-deployment-aks'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass 

aks_service = Model.deploy(workspace=ws,
                           name=service_name,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=aks_config,
                           deployment_target=aks_gpu_cluster)

#########################################################################################################

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)
