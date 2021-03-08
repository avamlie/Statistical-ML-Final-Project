# Statistical Machine Learning Final Project
### Annika Amlie and Robert Peng

## Reference Paper

Liheng Zhang, Charu Aggarwal, Guo-Jun Qi, Stock Price Prediction via Discovering Multi-Frequency Trading Patterns,
*Proceedings of ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (KDD 2017), Halifax, Nova Scotia,
Canada, August 13-17, 2017.

## Introduction

One of the goals of stock investors is to discover latent trading patterns in stock market to forecast future price trends and maximize profits. However, the prediction of stock prices using time series is a very challenging task, since they are affected by uncertain political-economic factors in the real world and are also non-stationary and non-linear. In addition, predicting stock prices in short or long-term time ranges relies on discovering different trading patterns in the security exchange market, ranging from low to high-frequency trading. Thus, explicitly discovering and separating various frequencies of latent trading patterns should play a crucial role in making price predictions for different ranges of time periods. 

While there have been many methods developed for stock price prediction, none explicitly decompose trading patterns into their various frequency components, and seamlessly integrate the discovered multi-frequency patterns into price predictions. The paper by Zhang, Aggarwal, and Qi develops such a method, a novel State Frequency Memory (SFM) recurrent network for stock price prediction, inspired by the Discrete Fourier Transform (DFT). In signal processing, the DFT is used to decompose input signals into multiple frequency components to study the periodicity and volatility of the signals; the same idea is applied to decompose the hidden states of trading patterns into multiple frequency components that will in turn drive the evolution of SFM over time. To deal with the non-linearity and non-stationarity of stock prices, an idea inspired by Long Short Term Memory (LSTM) is used, and its network structure allows it to capture the long-term dependency of stock prices at different times. The end result is the SFM recurrent network, which is able to learn multiple frequency trading patterns to make both short and long-term price predictions.

## Overview of Methods

The State Frequency Memory (SFM) recurrent network attempts to extract and leverage non-stationary trading patterns of multiple frequencies. Like  LSTM, SFM models the hidden states underlying a time series with a sequence of memory cells. However, the memory cells in the SFM consist of state components for multiple frequencies, each of which the authors call a *state-frequency component*. The following figure shows the block diagrams corresponding to RNN, LSTM, and SFM. 

![block diagrams](https://github.com/avamlie/Statistical-ML-Final-Project/blob/main/images/SFM%20Comparison.PNG?raw=true)

When Recurrent Neural Networks (RNN) are trained with Back-Propagation Through Time (BPTT), they generally suffer from vanishing gradients, making them unable to handle long-term dependency in a time series. The LSTM was proposed to address this problem by using additional gating units to maintain the long-term memory of the trading patterns from the historical prices.
