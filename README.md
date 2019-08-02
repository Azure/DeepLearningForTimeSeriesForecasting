# Deep Learning for Time Series Forecasting

A collection of examples for using DNNs for time series forecasting with Keras. The examples include:

- **0_data_setup.ipynb** - set up data that are needed for the experiments
- **1_CNN_dilated.ipynb** - dilated convolutional neural network model that predicts one step ahead with univariate time series
- **2_RNN.ipynb** - recurrent neural network model that predicts one step ahead with univariate time series
- **3_RNN_encoder_decoder.ipynb** - a simple recurrent neural network encoder-decoder approach to multi-step forecasting
- **4_ES_RNN.ipynb** - a simplified exponential smoothing recurrent neural network model that predicts one step ahead with univariate time series

... and a number of hands-on exercises and demos.


## Data

The data in all examples is from the GEFCom2014 energy forecasting competition<sup>1</sup>. It consists of 3 years of hourly electricity load data from the New England ISO and also includes hourly temperature data. Before running the notebooks, download the data from [https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0](https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0) and save it in the *data* folder.

<sup>1</sup>Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.


## Prerequisites

- Know the basics of Python
- Know the basics of neural networks
- Understand the basic concepts of machine learning and have some experience in building machine learning models
- Go through the following setup instructions to run notebooks on Azure Notebooks environment

*Note: If you want to run the notebook in other environment, please check 'Requirements.txt' for a list of packages that you need to install.*

### Azure Notebooks Setup (1 Minute)

Microsoft Azure Notebooks is a free service that provides Jupyter Notebooks in cloud and has a support for R, Python and F#. We will use Microsoft Azure Notebook for this tutorial. Here are quick 3 steps to set it up:

1. Go to Azure Notebook page https://notebooks.azure.com/ and click '***Sign In***' on the top right.
2. Use your Microsoft account to sign in. If you don't have a personal Microsoft account, you can click '***Create one***' with any email address you have for free. (*Note: You can use your personal Microsoft account. If you use your organizational account, you will need to go through the login process by your organization.*)
3. If this is the first time you use Azure Notebook, you will need to create a user ID and click '***Save***'. Now you are all set!


### Tutorial Code and Data Setup (5 - 10 Minutes)

The following steps will guide you to setup code and data in your Azure Notebook environment for the tutorial.

1. Once you are logged into Azure Notebooks, go to '***My Projects***' on the top left, and then click '***Upload GitHub Repo***'.

2. In the pop out window, for '***GitHub repository***' type in: '***Azure/DeepLearningForTimeSeriesForecasting***'. Select '***Clone recursively***'. Then type in any name you prefer for '***Project Name***' and '***Project ID***'. Once you have filled all boxes, click '***Import***'. Please wait till you see a list of files cloned from git repository to your project.

3. Open the notebook '***0_data_setup.ipynb***'. Make sure you see '***Python 3.6***' kernel on the top right. If not, you can select '***Kernel***', then '***Change kernel***' to make changes.

4. Run each cell in the notebook by click '***Run***' on top. If you prefer to run all the cells together, click '***Cell***' and select '***Run All***'. This notebook will download a sample dataset to your environment and visualize the data. Please wait and make sure you can see all the visualizations. 

Now you are all set! (*Note: If you see errors return from the first code cell, it is very likely that the environment preparation is not finished yet. Please wait for 2 minutes and then go to '***Kernel***', choose '***Restart and Clear Output***' and rerun the cells.*) 
