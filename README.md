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
![](https://latex.codecogs.com/gif.latex?%5Chat%20p_%7Bt%20&plus;%20n%7D%20%3D%20f%28p_t%2C%20p_%7Bt-1%7D%2C%20...%2C%20p_1%29), where f denotes the model mapping the historical prices to the price n-steps ahead. Since the scales of prices varies for different stocks, we normalize the prices of each stock to fall within [-1,1]. We adapt a RNN variant, the SFM or the LSTM, as such a mapping f for n-step prediction. The hidden vector 
![](https://latex.codecogs.com/gif.latex?h_t) from the RNN variant is used for price prediction through a matrix transformation. 

