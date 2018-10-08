# multi-step forecasting with recurrent neural network that generates vector output

The hyperparameters of recurrent neural network are tuned using Batch AI. To run this code in Linux:
* follow instructions in [setup.md](./setup.md) and provision Batch AI environment and related Azure services
* run
```bash
nohup python tune_RNN_multi_step_vector_output.py >& out.txt &
```

The running time depends on the size of your Batch AI cluster. With the default Batch AI quota (20 cores per account), the experiment finishes 
within 24 hours. When running with a cluster of 8 VMs of NC6 size, the experiment finishes within 8 hours.

To clean up a resource group containing all provisioned Azure services, run
```bash
az group delete -n <resource-group-name>     
az ad app delete --id www.<resource-group-name>.com
```

