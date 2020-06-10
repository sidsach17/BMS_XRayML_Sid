
from azureml.core import Workspace
import os, json, sys
import azureml.core
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.authentication import ServicePrincipalAuthentication


print("SDK Version:", azureml.core.VERSION)
# print('current dir is ' +os.curdir)
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

try:
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=az_sp
    )

except:
    # this call might take a minute or two.
    print("Creating new workspace")
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        # create_resource_group=True,
        location=location,
        auth=az_sp

    )

# print Workspace details
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")

#Creating Experiment

experiment_name = 'XRayML_AzureDevOps'

from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)