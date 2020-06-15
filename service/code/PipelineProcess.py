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
run_config_user_managed.environment.python.conda_dependencies = CondaDependencies.create(pip_packages = ['keras<=2.3.1','pandas','matplotlib',
                                                                                        'opencv-python','azure-storage-blob==2.1.0','tensorflow-gpu==2.0.0',
                                                                                        'azureml','azureml-core','azureml-dataprep',
                                                                                        'azureml-dataprep[fuse]','azureml-pipeline'])
#run_config_user_managed.environment.python.user_managed_dependencies = True

print("Pipeline SDK-specific imports completed")
#######################################################################################################
# Create CPU cluster for Data preprocessing
'''
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
'''
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
xrayimage_dataset = Dataset.get_by_name(ws, name='xray_image_ds')
traindata_dataset = Dataset.get_by_name(ws, name='train_data_ds')
validdata_dataset = Dataset.get_by_name(ws, name='valid_data_ds')
testdata_dataset = Dataset.get_by_name(ws, name='test_data_ds')
traintarget_dataset = Dataset.get_by_name(ws, name='train_target_ds')
validtarget_dataset = Dataset.get_by_name(ws, name='valid_target_ds')
testtarget_dataset = Dataset.get_by_name(ws, name='test_target_ds')

datastore = Datastore.get(ws,"xray_datastore")

PreProcessingData = PipelineData("PreProcessingData", datastore=datastore)
ModelData = PipelineData("ModelData", datastore=datastore)
####################################################################################################### 
preprocessing_step = PythonScriptStep(name="preprocessing_step",
                                      script_name="estimator_data_preprocessing.py", 
                                      compute_target=GPU_compute_target, 
                                      runconfig = run_config_user_managed,
                                      source_directory = './scripts/data_preprocess',
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

#######################################################################################################

est = TensorFlow(source_directory = './scripts/train', 
                    compute_target = GPU_compute_target,
                    entry_script = "estimator_training.py",
                    pip_packages = ['keras<=2.3.1','matplotlib','opencv-python','azure-storage-blob==2.1.0','tensorflow-gpu==2.0.0'],
                    conda_packages = ['scikit-learn==0.22.1'],
                    use_gpu = True )


est_step = EstimatorStep(name="Estimator_Train", 
                         estimator=est, 
                         estimator_entry_script_arguments=['--PreProcessingData', PreProcessingData],
                         inputs=[PreProcessingData],
                         runconfig_pipeline_params=None,
                         compute_target=GPU_compute_target)


#######################################################################################################

register_step = PythonScriptStep(name = "register_step",
                    script_name= "estimator_register.py",
                    runconfig = run_config_user_managed,
                    source_directory = './scripts/register',
                    arguments=['--ModelData', ModelData], 
                    outputs = [ModelData],
                    compute_target=GPU_compute_target 
                    )


#######################################################################################################
est_step.run_after(preprocessing_step)
register_step.run_after(est_step)

#Build Pipeline
pipeline = Pipeline(workspace = ws,steps=[preprocessing_step,est_step,register_step])

#Validate pipeline
pipeline.validate()
print("Pipeline validation complete")


#Publish the pipeline
published_pipeline = pipeline.publish(name="MLOps_Pipeline_Estimator", 
                                    description="MLOps pipeline for estimator",
                                    continue_on_step_failure=True)

#submit Pipeline
pipeline_run = exp.submit(pipeline,pipeline_parameters={})
print("Pipeline is submitted for execution")

#######################################################################################################
# Shows output of the run on stdout.
pipeline_run.wait_for_completion(show_output=True)

# Raise exception if run fails
if pipeline_run.get_status() == "Failed":
    raise Exception(
        "Training on local failed with following run status: {} and logs: \n {}".format(
            pipeline_run.get_status(), pipeline_run.get_details_with_logs()
        )
    )


# Writing the run id to /aml_config/run_id.json
'''
with open(os.path.join(ModelData, 'model.json')) as model_file:
    modeljson = json.load(model_file)
    model_json = {}
    model_json["model_name"] = modeljson['model_name']
    model_json["model_version"] = modeljson['model_version']

with open("./configuration/model.json", "w") as modelfile:
    json.dump(model_json, modelfile)
'''
run_id = {}
run_id["run_id"] = pipeline_run.id
run_id["experiment_name"] = pipeline_run.experiment.name
with open("./configuration/run_id.json", "w") as outfile:
    json.dump(run_id, outfile)


    
    