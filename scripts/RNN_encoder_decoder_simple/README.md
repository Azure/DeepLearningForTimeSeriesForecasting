# Multi-step forecasting using a simple encoder-decoder

The hyperparameters of encoder-decoder network are tuned using Batch AI. To run this code in Linux:
* follow instructions in [setup.md](./setup.md) and provision Batch AI environment and related Azure services
* run
```bash
nohup python tune_RNN_multi_step_encoder_decoder_simple.py >& out.txt &
```

The running time depends on the size of your Batch AI cluster. With the default Batch AI quota (20 cores per account), the experiment will run for several days. 
When running with a cluster of 10 VMs of NC6 size, the experiment finishes within 36 hours..

To clean up a resource group containing all provisioned Azure services, run
```bash
az group delete -n <resource-group-name>     
az ad app delete --id www.<resource-group-name>.com
```

