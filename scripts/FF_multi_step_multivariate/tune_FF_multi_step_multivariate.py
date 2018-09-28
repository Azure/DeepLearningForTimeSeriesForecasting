# multi-step forecasting with multivariate input using feed-forward neural network
# the hyperparameters of feed-forward neural network are tuned using Batch AI
# to use this code:
#     1. follow instructions in setup.md and provision Batch AI environment
#     2. copy configuration.json.template to configuration.json
#     3. fill all credentials and configuration parameters in configuration.json file
#########################################################################################################

import sys
import numpy as np
from azure.common.credentials import ServicePrincipalCredentials
import azure.mgmt.batchai as batchai
import azure.mgmt.batchai.models as models
import json

# load all credentials
configuration_file_name = "configuration.json"
with open(configuration_file_name, 'r') as f:
    cfg = json.load(f)

# load utilities of Batch AI
sys.path.append(cfg['utils_dir'])
import utilities as utils
from utilities.job_factory import ParameterSweep, NumericParameter, DiscreteParameter

# connect to Batch AI workspace and cluster
creds = ServicePrincipalCredentials(client_id=cfg['active_directory']['client_id'], secret=cfg['active_directory']['client_secret'], 
                                    tenant=cfg['active_directory']['tenant_id'])

batchai_client = batchai.BatchAIManagementClient(credentials=creds, subscription_id=cfg['subscription_id'])
cluster = batchai_client.clusters.get(cfg['resource_group'], cfg['batch_ai']['workspace'], cfg['batch_ai']['cluster_name'])

# define grid of tuned hyperparameters
param_specs = [
    DiscreteParameter(
        parameter_name="LATENT_DIM",
        values=[5,10,15]
    ),
    DiscreteParameter(
        parameter_name="HIDDEN_LAYERS",
        values=[1,2,3]
    ),
    DiscreteParameter(
        parameter_name="BATCH_SIZE",
        values=[8,16,32]
    ),
    DiscreteParameter(
        parameter_name="T",
        values=[72,168,336]
    ),
    DiscreteParameter(
        parameter_name="LEARNING_RATE",
        values=[0.01, 0.001, 0.0001]
    ),
     DiscreteParameter(
        parameter_name="ALPHA",
        values=[0.1,0.001,0]
    )
]

parameters = ParameterSweep(param_specs)

# create a template for Batch AI job
jcp = models.JobCreateParameters(
    cluster=models.ResourceId(id=cluster.id),
    node_count=1,
    std_out_err_path_prefix='$AZ_BATCHAI_JOB_MOUNT_ROOT/logs',
    output_directories=[
        models.OutputDirectory(
            id='ALL',
            path_prefix='$AZ_BATCHAI_JOB_MOUNT_ROOT/output'
        )
    ],
    custom_toolkit_settings=models.CustomToolkitSettings(
        command_line='python $AZ_BATCHAI_JOB_MOUNT_ROOT/resources/scripts/FF_multi_step_multivariate.py \
        --scriptdir $AZ_BATCHAI_JOB_MOUNT_ROOT/resources/scripts \
        --datadir $AZ_BATCHAI_JOB_MOUNT_ROOT/resources/data \
        --outdir $AZ_BATCHAI_OUTPUT_ALL \
        -l {0} -n {1} -b {2} -T {3} -r {4} -a {5}'.format(parameters['LATENT_DIM'], parameters['HIDDEN_LAYERS'],
                                                          parameters['BATCH_SIZE'], parameters['T'],
                                                          parameters['LEARNING_RATE'], parameters['ALPHA'])
    ),
    container_settings=models.ContainerSettings(
        image_source_registry=models.ImageSourceRegistry(image=cfg['docker_image'])
    ),
    mount_volumes = models.MountVolumes(
        azure_file_shares=[
            models.AzureFileShareReference(
                account_name=cfg['storage_account']['name'],
                credentials=models.AzureStorageCredentialsInfo(account_key=cfg['storage_account']['key']),
                azure_file_url='https://'+cfg['storage_account']['name']+'.file.core.windows.net/logs',
                relative_mount_path='logs'),
            models.AzureFileShareReference(
                account_name=cfg['storage_account']['name'],
                credentials=models.AzureStorageCredentialsInfo(account_key=cfg['storage_account']['key']),
                azure_file_url='https://'+cfg['storage_account']['name']+'.file.core.windows.net/resources',
                relative_mount_path='resources'),
            models.AzureFileShareReference(
                account_name=cfg['storage_account']['name'],
                credentials=models.AzureStorageCredentialsInfo(account_key=cfg['storage_account']['key']),
                azure_file_url='https://'+cfg['storage_account']['name']+'.file.core.windows.net/output',
                relative_mount_path='output'),
        ]
    )
)

# submit many jobs - one job per combination of values of hyperparameters
jobs_to_submit, param_combinations = parameters.generate_jobs(jcp)
experiment_utils = utils.experiment.ExperimentUtils(batchai_client, cfg['resource_group'], 
                                                    cfg['batch_ai']['workspace'], cfg['batch_ai']['experiment'])
jobs = experiment_utils.submit_jobs(jobs_to_submit, cfg['batch_ai']['job_name_prefix']).result()

# define extractors of metric values
validation_mape_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt',
                                                      regex='Mean validation MAPE = (.*?) ')
validation_se_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt',
                                                    regex='Mean validation MAPE = ...... \+\/\- (.*)\\nMean ')
test_mape_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt',
                                                regex='Mean test MAPE = (.*?) ')
test_se_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt',
                                              regex='Mean test MAPE = ...... \+\/\- (.*)')
T_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt', regex='T=(.*), LA')
latent_dim_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt', 
                                                 regex='LATENT_DIM=(.*), HI')
hidden_layers_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt', 
                                                    regex='HIDDEN_LAYERS=(.*), BA')
batch_size_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt', 
                                                 regex='BATCH_SIZE=(.*), LR')
learning_rate_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt', 
                                                    regex='LR=(.*), AL')
alpha_extractor = utils.job.MetricExtractor(output_dir_id='ALL', logfile='output.txt', 
                                            regex='ALPHA=(.*?)\\n')

# wait for all jobs to complete
experiment_utils.wait_all_jobs()

# extract metric and hyperparameters values of each job
validation_mape_recs = experiment_utils.get_metrics_for_jobs(jobs, validation_mape_extractor)
validation_se_recs = experiment_utils.get_metrics_for_jobs(jobs, validation_se_extractor)
test_mape_recs = experiment_utils.get_metrics_for_jobs(jobs, test_mape_extractor)
test_se_recs = experiment_utils.get_metrics_for_jobs(jobs, test_se_extractor)
T_recs = experiment_utils.get_metrics_for_jobs(jobs, T_extractor)
latent_dim_recs = experiment_utils.get_metrics_for_jobs(jobs, latent_dim_extractor)
hidden_layers_recs = experiment_utils.get_metrics_for_jobs(jobs, hidden_layers_extractor)
batch_size_recs = experiment_utils.get_metrics_for_jobs(jobs, batch_size_extractor)
learning_rate_recs = experiment_utils.get_metrics_for_jobs(jobs, learning_rate_extractor)
alpha_recs = experiment_utils.get_metrics_for_jobs(jobs, alpha_extractor)

def extract_metric_value(l):
    return [x['metric_value'] for x in l]

# create arrays of metric and hyperparparameter values
validation_mape = extract_metric_value(validation_mape_recs)
validation_se = extract_metric_value(validation_se_recs)
test_mape = extract_metric_value(test_mape_recs)
test_se = extract_metric_value(test_se_recs)
T = extract_metric_value(T_recs)
latent_dim = extract_metric_value(latent_dim_recs)
hidden_layers = extract_metric_value(hidden_layers_recs)
batch_size = extract_metric_value(batch_size_recs)
learning_rate = extract_metric_value(learning_rate_recs)
alpha = extract_metric_value(alpha_recs)

# choose hyperparameters that minimize validation MAPE
best_run = np.argmin(validation_mape)

# print MAPE of the chosen model
out = 'Validation MAPE {0} +/- {1}, Test MAPE {2} +/- {3}'.format(validation_mape[best_run], validation_se[best_run],
                                                                  test_mape[best_run], test_se[best_run])
print(out)

# print chosen values of hyperparameters
params = 'T={0} LATENT_DIM={1} HIDDEN_LAYERS={2} BATCH_SIZE={3} LEARNING_RATE={4} ALPHA={5}'.format(T[best_run],
             latent_dim[best_run], hidden_layers[best_run], batch_size[best_run], learning_rate[best_run], alpha[best_run])
print(params)
