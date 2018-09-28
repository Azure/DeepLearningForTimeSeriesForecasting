# Environment Setup

This code uses Azure service to run training process remotely. To get a free Azure account, please follow instructions [here](https://azure.microsoft.com/en-us/free/).
We use Azure CLI to set up the environment. To install the latest version of Azure CLI, please follow instructions [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest).

* Log into Azure account by running in Bash shell:
```bash
az login
```
and following instructions there. If your account is linked to multiple subscriptions, find the id of the subscription you want to use by running
```bash
az account list --all -o table
```
and set this subscription as active by running
```bash
az account set -s <subscription-id>
```
If your Azure account is linked to a single subcription, get your subscription id by running
```bash
az account show | grep "id"
```
* Download the dataset
```bash
wget https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip
```
* Create Batch AI resources by running
```bash
./setup_batchai -s <subscription_id> -r <azure-region> -g <resource-group>
```
where     
    - \<resource-group\> is the name of Azure resource group where all resources will be created. If the resource group does not exist then it will be created by the script. We recommend to use a dedicated resource group for this experiment.    
    - \<azure-region\> is Azure region where all resources will be created, one of "eastus", "eastus2", "southcentralus", "westcentralus", "westus2".

