# Stock-Price-Prediction
The art of forecasting stock prices has been a difficult task for many of the researchers and analysts. In fact, investors are highly interested in the research area of stock price prediction.
# Stock Price Prediction Using LSTM

This project predicts stock prices using a Long Short-Term Memory (LSTM) neural network. It processes historical stock data to forecast future closing prices, providing insights for financial analysis.

## Features
- Visualizes historical stock price trends.
- Preprocesses data using normalization and sequence creation.
- Implements an LSTM model to predict stock prices.
- Provides visual comparisons between actual and predicted prices.

## Prerequisites
The following dependencies are required to run this program:
- Python 3.7+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>

## Usage
1.  Edit the file_path variable in the script to point to your dataset:
file_path = "stock_data.csv"
2.  The dataset should include a Date column and a Close column for closing prices.
Run the script:
python Stock_prediction.py
