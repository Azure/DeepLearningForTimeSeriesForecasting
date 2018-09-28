# multi-step forecasting with multivariate input using feed-forward neural network

The hyperparameters of feed-forward neural network are tuned using Batch AI. To run this code in Linux:
* follow instructions in [setup.md](./setup.md) and provision Batch AI environment and related Azure services
* run
```bash
nohup python tune_FF_multi_step_multivariate.py >& out.txt &
```

The running time depends on the size of your Batch AI cluster. With the default Batch AI quota (20 cores per account), the experiment finishes 
within 24 hours. When running with a cluster of 8 VMs of NC6 size, the experiment finishes within 8 hours.

To clean up a resource group containing all provisioned Azure services, run
```bash
az group delete -n <resource-group-name>
```

