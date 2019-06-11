# Hyper-parameter tuning using Azure Machine Learning service

The hyper-parameters of methods presented in this tutorial are tuned using Hyperdrive, a feature of Azure Machine Learning (Azure ML) service. 


### Pre-Requisites

- You'll need to first set up the environment using
[Anaconda3](https://www.anaconda.com/distribution/#download-section). To do so, navigate to the `hyperparameter_tuning` directory and run the following commands:

    ```bash
    # create a conda environment using the environment.yaml file
    # replace 'create' with 'update' if you need to update the environment
    conda env create -f environment.yaml
    source activate dnntutorial
    # to install the environment into the Jupyter kernel, run
    python -m ipykernel install --user --name dnntutorial
    ```

- Next, please follow instructions in [configuration notebook](../configuration.ipynb) in order to provision Azure ML workspace.

> Note: This material is developed on an [Azure Data Science Virtual Machine (Ubuntu)](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/overview). This machine is pre-installed with many development tools we use in this tutorial, such as Anaconda, Python, etc.

### Run hyper-parameter tuning notebook

Hyper-parameter tuning is done in [hyperparameter_tuning.ipynb](./hyperparameter_tuning.ipynb) notebook. This notebook is used to tune several approaches:
- Feed-forward network multi-step multivariate - [ff_multistep_config.json](ff_multistep_config.json)
- RNN multi-step - [rnn_multistep_config.json](rnn_multistep_config.json)
- RNN teacher forcing - [rnn_teacher_forcing_config.json](rnn_teacher_forcing_config.json)
- RNN encoder decoder - [rnn_encoder_decoder_config.json](rnn_encoder_decoder_config.json)
- CNN - [cnn_config.json](cnn_config.json)

Each of these use cases is defined in a json configuration file listed above alongside each usecase. To run a specific approach, please specify the appropriate configuration file in the hyperparameter_tuning notebook.


The running time depends on the size of your Azure ML cluster and the model being tuned.
