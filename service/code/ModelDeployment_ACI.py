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

###################################################################################################
'''
with open("./configuration/model.json") as modelinput:
    model_json = json.load(modelinput)

model_name=model_json["model_name"]
model_version=model_json["model_version"]
'''
model_name="MLOps_Model"
#model = Model(workspace=ws, name=model_name, version=model_version)
model = Model(workspace=ws, name=model_name)

print("model name: ",model_name)

###################################################################################################

deployment_env = Environment.from_conda_specification(name="deployment_env", file_path="./configuration/deployment_env.yml")
inference_config = InferenceConfig(entry_script="./scripts/score/score.py", environment=deployment_env)

 

aciconfig = AciWebservice.deploy_configuration(cpu_cores=2,
                                               auth_enabled=True, # this flag generates API keys to secure access
                                               memory_gb=8,
                                               #tags={'name': 'mnist', 'framework': 'Keras'},
                                               description='X-Ray ML Estimator ACI endpoint')

###################################################################################################

service_name = 'mlops-estimator-model-aci'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass 

service = Model.deploy(workspace=ws, 
                           name=service_name, 
                           models=[model], 
                           inference_config=inference_config, 
                           deployment_config=aciconfig)

###################################################################################################

service.wait_for_deployment(True)
print(service.state)

print(service.scoring_uri)
'''
score_uri = {}
score_uri["scoring_uri"] = service.scoring_uri

with open("./configuration/scoring_uri_aci.json", "w") as outfile:
    json.dump(score_uri, outfile)
'''