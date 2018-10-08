# Multi-step forecasting with encoder-decoder network and teacher forcing

The hyperparameters of encoder-decoder network are tuned using Batch AI. To run this code in Linux:
* follow instructions in [setup.md](./setup.md) and provision Batch AI environment and related Azure services
* run
```bash
nohup python tune_RNN_teacher_forcing.py >& out.txt &
```

The running time depends on the size of your Batch AI cluster. With the default Batch AI quota (20 cores per account), the experiment runs for several days. When running 
with a cluster of 10 VMs of NC6 size, the experiment finishes within 12 hours.

To clean up a resource group containing all provisioned Azure services, run
```bash
az group delete -n <resource-group-name>     
az ad app delete --id www.<resource-group-name>.com
```

