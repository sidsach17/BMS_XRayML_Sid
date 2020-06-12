from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import json

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
    
    
#Creating CPU cluster
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


#Creating GPU cluster
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