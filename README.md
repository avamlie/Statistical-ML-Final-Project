# Statistical Machine Learning Final Project
### Annika Amlie and Robert Peng

## Reference Paper

Liheng Zhang, Charu Aggarwal, Guo-Jun Qi, Stock Price Prediction via Discovering Multi-Frequency Trading Patterns,
*Proceedings of ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (KDD 2017), Halifax, Nova Scotia,
Canada, August 13-17, 2017.

## Introduction

One of the goals of stock investors is to discover latent trading patterns in stock market to forecast future price trends and maximize profits. However, the prediction of stock prices using time series is a very challenging task, since they are affected by uncertain political-economic factors in the real world and are also non-stationary and non-linear. In addition, predicting stock prices in short or long-term time ranges relies on discovering different trading patterns in the security exchange market, ranging from low to high-frequency trading. Thus, explicitly discovering and separating various frequencies of latent trading patterns should play a crucial role in making price predictions for different ranges of time periods. 

While there have been many methods developed for stock price prediction, none explicitly decompose trading patterns into their various frequency components, and seamlessly integrate the discovered multi-frequency patterns into price predictions. The paper by Zhang, Aggarwal, and Qi develops such a method, a novel State Frequency
Memory (SFM) recurrent network for stock price prediction, inspired by the Discrete Fourier Transform (DFT). 
