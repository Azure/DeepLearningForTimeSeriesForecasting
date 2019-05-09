# Hyper-parameter tuning using Azure Machine Learning service

The hyper-parameters of methods presented in this tutorial are tuned using Hyperdrive, a feature of Azure Machine Learning (Azure ML) service. 

> NOTE: Before running this notebook, please follow instructions in [configuration notebook](../configuration.ipynb) and provision Azure ML workspace.

The running time depends on the size of your Azure ML cluster and the method being tuned.

Hyper-parameter tuning is done in [hyperparameter_tuning.ipynb](./hyperparameter_tuning.ipynb) notebook. This notebook is used to tune several approaches:
- Feed-forward network multi-step multivariate approach - [ff_multistep_config.json](ff_multistep_config.json)
- RNN multi-step approach - [rnn_multistep_config.json](rnn_multistep_config.json)
- RNN teacher forcing approach - [rnn_teacher_forcing_config.json](rnn_teacher_forcing_config.json)
- RNN encoder decoder approach - [rnn_encoder_decoder_config.json](rnn_encoder_decoder_config.json)

Each of these use cases is defined in a json configuration file listed above alongside each usecase. To run a specific approach, please specify the appropriate configuration file in the hyperparameter_tuning notebook.