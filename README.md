# Stock Market Prediction with Swing Points and Random Forest

This repository contains a machine learning algorithm that predicts stock market movement (whether the market will go up or down) using swing points (highs and lows) and other technical indicators from historical market data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Fetching](#data-fetching)
- [Machine Learning Model](#machine-learning-model)
- [Example Output](#example-output)
- [License](#license)

## Overview

This project uses a **Random Forest Classifier** to predict the market's daily bias (up or down) based on historical stock market data. The algorithm analyzes swing highs and lows over a customizable period and incorporates whether these swing points have been raided (i.e., surpassed by another high/low) to generate additional predictive features.

### What are Swing Points?

- **Swing High**: A price point that is higher than the prices around it.
- **Swing Low**: A price point that is lower than the prices around it.
- These points are used to determine key support and resistance levels in the market, which can signal future price movements.

## Features

- Fetches historical market data using `yfinance`.
- Identifies and filters **swing highs** and **swing lows**.
- Determines whether swing points have been **unliquidated** (not surpassed by subsequent highs/lows).
- Implements a **Random Forest Classifier** to predict whether the market will move up (green) or down (red) based on the features.
- Adds functionality to make **daily predictions** and retrain the model with new data.

## Installation

### Prerequisites

Make sure you have Python 3.x installed, and install the following dependencies:


pip install yfinance
pip install scikit-learn
pip install pandas
pip install numpy

Machine Learning Model
The model uses a Random Forest Classifier from scikit-learn. It is trained on various features, such as open price, high price, low price, close price, volume, swing highs/lows, and whether those swing points were raided.

The model is retrained each day with the new data, improving its accuracy over time.

Features Used:
Open
High
Low
Close
Volume
SwingHigh (binary: 1 if there was a swing high, 0 otherwise)
SwingLow (binary: 1 if there was a swing low, 0 otherwise)
SwingHigh_Raided (binary: 1 if the swing high was raided, 0 otherwise)
SwingLow_Raided (binary: 1 if the swing low was raided, 0 otherwise)


This project is licensed under the MIT License.
