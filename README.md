# Recurrent Neural Networks for Time Series Forecasting

A collection of examples for using RNNs for time series forecasting with Keras. The examples include:

- **1_one_step_univariate.ipynb** - model that predicts one step ahead with univariate time series
- **2_one_step_multivariate.ipynb** - example of using multivariate time series data
- **3_multi_step_vector_output.ipynb** - model that outputs a vector of predictions to forecast multiple steps ahead
- **4_multi_step_encoder_decoder_simple.ipynb** - a simple encoder-decoder approach to multi-step forecasting
- **5_multi_step_encoder_decoder_teacher_forcing.ipynb** - a more complex encoder-decoder architecture in which the decoder is trained using a teacher forcing approach

The data in all examples is from the GEFCom2014 energy forecasting competition<sup>1</sup>. It consists of 3 years of hourly electricity load data from the New England ISO and also includes hourly temperature data. Before running the notebooks, download the data from [https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0](https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0) and save it in the *data* folder.

## Prerequisites

You must have the following software and packages installed to run these notebooks:
- Anaconda
- Keras

<sup>1</sup>Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.