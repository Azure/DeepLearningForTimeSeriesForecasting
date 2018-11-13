# Recurrent Neural Networks for Time Series Forecasting

A collection of examples for using RNNs for time series forecasting with Keras. The examples include:

- **0_data_setup.ipynb** - set up data that are needed for the experiments
- **1_time_series_arima.ipynb** - ARIMA model for time series forecasting
- **2_one_step_FF_univariate.ipynb** - feed forward neural network model that predicts one step ahead with univariate time series
- **3_one_step_RNN_univariate.ipynb** - recurrent neural network model that predicts one step ahead with univariate time series
- **4_multi_step_RNN_vector_output.ipynb** - recurrent neural network model that outputs a vector of predictions to forecast multiple steps ahead
- **5_multi_step_RNN_encoder_decoder_simple.ipynb** - a simple recurrent neural network encoder-decoder approach to multi-step forecasting


## Data

The data in all examples is from the GEFCom2014 energy forecasting competition<sup>1</sup>. It consists of 3 years of hourly electricity load data from the New England ISO and also includes hourly temperature data. Before running the notebooks, download the data from [https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0](https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0) and save it in the *data* folder.

<sup>1</sup>Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.


## Prerequisites

- Know the basics of Python
- Understand the basic concepts of machine learning and have some experience in building machine learning models
- Go through the following setup instructions to run notebooks on Azure Notebooks environment

*Note: If you want to run the notebook in other environment, please check 'Requirements.txt' for a list of packages that you need to install.*

### Azure Notebooks Setup (1 Minute)

Microsoft Azure Notebooks is a free service that provides Jupyter Notebooks in cloud and has a support for R, Python and F#. We will use Microsoft Azure Notebook for this tutorial. Here are quick 3 steps to set it up:

1. Go to Azure Notebook page https://notebooks.azure.com/ and click '***Sign In***' on the top right.
2. Use your Microsoft account to sign in. If you don't have a personal Microsoft account, you can click '***Create one***' with any email address you have for free. (*Note: You can use your personal Microsoft account. If you use your organizational account, you will need to go through the login process by your organization.*)
3. If this is the first time you use Azure Notebook, you will need to create a user ID and click '***Save***'. Now you are all set!


### Tutorial Code and Data Setup (5 Minutes)

The following steps will guide you to setup code and data in your Azure Notebook environment for the tutorial. Note: the code repository will be open one week before the conference, please do this setup after October 1st, 2018.

1. Once you are logged in to Azure Notebooks, go to '***Libraries***' on the top left, and then click '**_+ New Library_**'.
2. In the pop out window, select '***From GitHub***' tab, and in '***GitHub repository***' type in: '***Azure/RNNForTimeSeriesForecasting***'. Then type in any name you prefer for '***Library Name***' and '***Library ID***'. Once you have filled in all boxes, click '***Import***'. Wait till you see a list of files cloned from git repository to your library.
3. Now let's set up your notebook environment. Click on '***Settings***'. In the pop-out window choose '***Environment***' tab, then select '**_+ Add_**'. Click on the drop-down menu '***Select Operation***' and choose '***Requirements.txt***'. For '***Select Target File***' choose '***Requirements.txt***'. For '***Select Python Version***' choose '***Python Version 3***'. Lastly click on '***Save***'.
4. When you are back to your library, click on '***0_data_setup.ipynb***'. It will take about 2 minutes for the Azure Notebooks to set up your environment. Please wait for 2 mintues before you run the notebook.
5. Make sure you see '***Python 3***' kernel on the top right. If not, you can select '***Kernel***', then '***Change kernel***' to make changes.
6. Run each cell in the notebook by click '***Run***' on top. This notebook will download sample data to your environment and visualize the data. Wait and make sure you can see all the visualizations. Now you are all set! (*Note: If you see errors return from the first code cell, it is very likely that the environment preparation is not finished yet. Please wait for 2 minutes and then go to '***Kernel***', choose '***Restart and Clear Output***' and rerun the cells.*) 
