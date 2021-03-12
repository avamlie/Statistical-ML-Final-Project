# Statistical Machine Learning Final Project
### Annika Amlie and Robert Peng

## Reference Paper

Liheng Zhang, Charu Aggarwal, Guo-Jun Qi, Stock Price Prediction via Discovering Multi-Frequency Trading Patterns,
*Proceedings of ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (KDD 2017), Halifax, Nova Scotia,
Canada, August 13-17, 2017.

Github page corresponding to paper: https://github.com/z331565360/State-Frequency-Memory-stock-prediction

## Introduction

One of the goals of stock investors is to discover latent trading patterns in stock market to forecast future price trends and maximize profits. However, the prediction of stock prices using time series is a very challenging task, since they are affected by uncertain political-economic factors in the real world and are also non-stationary and non-linear. In addition, predicting stock prices in short or long-term time ranges relies on discovering different trading patterns in the security exchange market, ranging from low to high-frequency trading. Thus, explicitly discovering and separating various frequencies of latent trading patterns should play a crucial role in making price predictions for different ranges of time periods. 

While there have been many methods developed for stock price prediction, none explicitly decompose trading patterns into their various frequency components, and seamlessly integrate the discovered multi-frequency patterns into price predictions. The paper by Zhang, Aggarwal, and Qi develops such a method, a novel State Frequency Memory (SFM) recurrent network for stock price prediction, inspired by the Discrete Fourier Transform (DFT). In signal processing, the DFT is used to decompose input signals into multiple frequency components to study the periodicity and volatility of the signals; the same idea is applied to decompose the hidden states of trading patterns into multiple frequency components that will in turn drive the evolution of SFM over time. To deal with the non-linearity and non-stationarity of stock prices, an idea inspired by Long Short Term Memory (LSTM) is used, and its network structure allows it to capture the long-term dependency of stock prices at different times. The end result is the SFM recurrent network, which is able to learn multiple frequency trading patterns to make both short and long-term price predictions.

## Methods

### Comparison with Prior Methods

The State Frequency Memory (SFM) recurrent network attempts to extract and leverage non-stationary trading patterns of multiple frequencies. Like  LSTM, SFM models the hidden states underlying a time series with a sequence of memory cells. However, the memory cells in the SFM consist of state components for multiple frequencies, each of which the authors call a *state-frequency component*. The following figure shows the block diagrams corresponding to RNN, LSTM, and SFM, giving an overview of the three methods:

![block diagrams](https://github.com/avamlie/Statistical-ML-Final-Project/blob/main/images/SFM%20Comparison.PNG?raw=true)

When Recurrent Neural Networks (RNN) are trained with Back-Propagation Through Time (BPTT), they generally suffer from vanishing gradients, making them unable to handle long-term dependency in a time series. The LSTM was proposed to address this problem by using additional gating units to maintain the long-term memory of the trading patterns from the historical prices. Three types of gating units control the stock-trading information entering and leaving a memory cell at each time: the input, forget, and output gates. The gating architecture of the LSTM allows it to achieve a balance between short and long term dependencies over the stock prices in a time series.

### The State Frequency Memory (SFM)

Building on the LSTM and inspired by the Discrete Fourier Transform (DFT), the authors propose the State Frequency Memory (SFM) recurrent network to enable the discovery and modeling of latent trading patterns across multiple frequency bands underlying the fluctuation of stock prices. The SFM models the dynamics of an input time series 
![equation](https://latex.codecogs.com/gif.latex?%5C%7B%20x_t%20%7C%20t%20%3D%201%2C%20...%2C%20T%20%5C%7D) with a sequence of memory cells. The memory states of SFMs are decomposed into a set of K evenly-spaced discrete frequencies. This constructs a joint decomposition of states and frequencies to capture the temporal context of the input time series. The memory states of the SFM is represented as a matrix 
![](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bs%7D_t) at time t, with rows and columns corresponding to D states and K frequencies. Then, the state-frequency memory of the SFM evolves as a combination of the gated past memory and the current input, like the LSTM. In contrast to the LSTM, the recurrent update of the SFM depends on different state-frequency components, reflecting the goal to make a short or a long-term prediction over stock prices. 

The memory gating architecture is used to balance the short and long-term dependency in a time series. The input gate regulates the amount of new information that can flow into the current memory cell, and a joint state-frequency forget gate matrix controls how much information on states and frequencies should be kept in the memory cell. The input modulation aggregates the current input information, which is then decomposed into a set of frequency bands. This multi-frequency decomposition of input information allows the SFM to discover trading patterns across several frequencies. A state forget gate and a frequency forget gate regulate how much of the past information should be kept in the memory cell. They are combined into a state-frequency forget gate to jointly regulate the state and frequency information. To obtain the output hidden state 
![](https://latex.codecogs.com/gif.latex?h_t), a state-only memory state is reconstructed to aggregate the information over various frequencies on the state amplitude, in a manner similar to the Inverse Fourier Transformation (IFT), which recovers the original signal by combining frequency components. However, rather than using the standard IFT basis, the model learns the weights for objective learning tasks.

Since the SFM is a variant of recurrent neural networks, it can also be trained using the the BPTT algorithm. The SFM recurrent network keeps the gating architecture of the LSTM, so it retains the ability to capture the long term dependency of time series. Furthermore, the specialized complex-valued memory states can model multiple patterns with different frequencies. For short term prediction, more high-frequency components are needed to capture the high volatility. 

### Price Prediction using the SFM

Since both the SFM and the LSTM are variants of recurrent neural networks, they can be used to predict stock prices as shown in the following diagram: 

![price prediction](https://github.com/avamlie/Statistical-ML-Final-Project/blob/main/images/price_prediction.PNG?raw=true)

The RNN cell in the middle can be either a LSTM cell or a SFM cell. The LSTM-based price prediction model is considered to be the baseline which will be used to compare against the proposed SFM model. For a time series of trading prices 
![](https://latex.codecogs.com/gif.latex?%5C%7Bp_t%20%7C%20t%20%3D%201%2C%20...%2C%20T%20%5C%7D) of a stock, we wish to make an n-step prediction for time t+n based on prices up to time t. The prediction can be viewed as a function 
![](https://latex.codecogs.com/gif.latex?%5Chat%20p_%7Bt%20&plus;%20n%7D%20%3D%20f%28p_t%2C%20p_%7Bt-1%7D%2C%20...%2C%20p_1%29), where f denotes the function mapping the historical prices to the price n-steps ahead. Since the scales of prices varies for different stocks, we normalize the prices of each stock to fall within [-1,1]. We adapt the RNN variant, either the SFM or the LSTM, as the mapping f for the n-step prediction. The hidden vector 
![](https://latex.codecogs.com/gif.latex?h_t) from the chosen RNN variant is used for price prediction through a matrix transformation. The RNN layer is flexible and can be replaced by any RNN variant; the two methods used in the paper are the SFM and the LSTM (for comparison). 

To learn general trading patterns from the stock market, the prices of multiple stocks are used to train the regression network, by minimizing the sum of squared errors between the predicted and true normalized prices in the training set (described later). All model parameters are updated through the BPTT algorithm. 

## Data 

The data used in the paper and in this project is stock prices retrieved from Yahoo! Finance; in particular, it consists of the daily opening prices of 50 stocks among ten sectors from 2007 to 2016. For each of the sectors, corporations with the top 5 market capitalization are selected. There are 2518 days of historical stock price data. All models are trained using the daily prices consisting of 2014 days from 2007 to 2014, and daily prices during 2015 and 2016 are used to validate and test respectively, both consisting of 252 days. With daily prices corresponding to 50 corporations over a time span of almost 10 years, this should be enough for the models to learn general patterns in the market over time.

The datasets corresponding to the paper can be [found here](https://github.com/z331565360/State-Frequency-Memory-stock-prediction/tree/master/dataset/price_long_50), consisting of 50 CSV files. 

### Results of Experiments

When the authors compared the SFM method with both autoregressive (AR) models and LSTM, they found that the SFM outperforms both the AR and the LSTM models in terms of average square error for 1-step, 3-step and 5-step predictions:

![Model Comparison](https://github.com/avamlie/Statistical-ML-Final-Project/blob/main/images/model_comparison.PNG?raw=true)

As seen in the table above, the performance of each of the models becomes worse as the prediction step increases, as expected, but the SFM performed the best for all prediction steps. Since LSTM and SFM are trained with the same procedure (the RMSprop optimizer with a fixed learning rate of 0.01), their performance can be directly compared, and we can see that the SFM seems to have a more precise prediction than the LSTM, which only models internal dependencies. The better performance of the SFM is likely due to the fact that SFM filters out irrelevant frequency components and retains relevant components to better predict future trend, preventing states from dominating price predictions by trapping the model into local patterns of price changes.

## Running Code for SFM Recurrent Network Model
### Setup
In order to run the model, we must ensure that Python 2.7, Keras 1.0.1, and Theano 0.9 are installed. Here is an example of how to install the proper version:
`pip install keras==1.0.1`

Additionally, we must clone the repository: `git clone https://github.com/z331565360/State-Frequency-Memory-stock-prediction.git`
### Preparing Data
The build_data.py file prepares the provided data by reading each of the CSV (in this case) files and grabbing the data from the proper column. In the Github provided above, the authors used Yahoo! Finance stock price data, where "Open" was the header of the column data needed for the prediction. In the following code, the data from this column is transposed into an array that will be later used to train and test the model.

```
all_data = np.zeros((len(filenames),2518),dtype = np.float32)
for i in range(len(filenames)):
  filename = filenames[i]
  print(i)
  print(filename)
  
  data=pd.read_csv(directory+'/'+filename)
  vars = ['Open']
  data = data[vars]
  data = np.array(data)
  data = np.transpose(data)
  data = data[0]
  data = data[::-1]
  
  all_data[i] = data

print(all_data.shape)  
np.save('data',all_data)
```
The 2518 from the first line of code represents the length of each CSV file from their data. If we want to replicate this with new data, we will have to preprocess the files to all be of the same length. 

In Linux, we run the following commands:
```
cd dataset
python build_data.py
```
### Testing with Pre-trained Model
In the test folder, we run `python test.py --step=1`
The number step we use indicates the n-step prediction model we want to use. The 1, 3, and 5 step predictions are provided in the source code.

<img src="https://github.com/avamlie/Statistical-ML-Final-Project/blob/main/images/test_output.png" width="600" height="200">

Additionally, we can visualize the data by running the command `python test --step=1 --visualization=true`, which produces several graphs:

<img src="https://github.com/avamlie/Statistical-ML-Final-Project/blob/main/images/visualization.png" width="400" height="300">
Visualization for AAPL data.

These visualizations show the similarity between the true, given data and the prediction model.

### Training
In the training folder, we run `python train.py --step=3 --hidden_dim=50 --freq_dim=10 --niter=4000 --learning_rate=0.01`, where `step` refers to the n-step prediction, `hidden_dim` and `freq_dim` refer to the dimensions, and `niter` (num iterations) and `learning_rate` correspond to the specific training parameters.

<img src="https://github.com/avamlie/Statistical-ML-Final-Project/blob/main/images/training-final.png" width="600" height="400">

The epochs shown in this screenshot represent the iterations of training. The output also includes training error, value error, training duration, best iteration and smallest error. These parameters can be altered depending on the desired accuracy of the model.

## Historical Prices and New Articles Data
In order to run the model with the provided Historical Stock Prices data and the News Articles data, we had to perform some data preprocessing. 

Since the data from the original repository had been written for CSVs, the historical prices data fit fairly well with the build_data code. However, the original CSV files were all the same length, so we needed to split the CSVs into files of the same length (preferably 2518, which was the length of the original files). To do this, we used the following split function (after making some alterations):

```
def split(filehandler, delimiter=',', row_limit=1000,
          output_name_template='output_%s.csv', output_path='.', keep_headers=True):
    import csv
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = reader.next()
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)
```
Source: https://gist.github.com/jrivero/1085501

Additionally, we had to change some of the code in build_data.py to account for the difference in column names. In the original data, values from the 'Open' columns were used, but in the historical prices data, the column name was different. Once we solved these smaller issues, the model was able to train and test the prediction.

Since the news articles data was originally in json format, we had to use a json -> csv converter: https://json-csv.com/
After converting the data, we followed the same steps as above and ran the model.
