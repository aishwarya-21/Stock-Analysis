# Stock-Analysis
This project focuses on predicting stock prices using a combination of Long Short-Term Memory (LSTM) and XGBoost models, with a fully deployable web application created using Streamlit. The project leverages historical stock market data to forecast future trends and provide actionable insights.

**Project Overview**

Objective: Predict stock prices for the next 30 days based on historical data.

Tech Stack: Python, Pandas, NumPy, Matplotlib, Seaborn, Sklearn, TensorFlow/Keras, XGBoost, Streamlit.

Deployment: Streamlit web app.

**Files & Directories**

Notebook (.ipynb file): Contains the complete code for data preprocessing, EDA, feature engineering, model building, evaluation, and hyperparameter tuning.

Model File (.h5): Saved LSTM model for deployment.

Stock App (app.py): Streamlit application to showcase interactive stock price forecasting.

AAPL CSV File: Historical Apple stock data used for training and testing the model.

**Project Workflow**

Data Collection:

Retrieved historical Apple stock data (AAPL.csv).

Time-based train-test split (2010-2018 for training, 2019 for testing).

Exploratory Data Analysis (EDA):

Visualized price trends, trading volume, and returns.

Identified patterns like volatility and seasonal trends.

Feature Engineering:

Created lag features, moving averages (7-day & 30-day), and volatility indicators.

Model Development:

Built an LSTM model to capture temporal dependencies.

Used XGBoost for feature importance and benchmarking.

Model Evaluation:

Evaluated models using MAE, RMSE, and R².

Deployment:

Developed a Streamlit web app to visualize stock price predictions.

Integrated the LSTM model with the web app using the saved .h5 file.


**Results**

Accurate short-term stock price predictions.

Interactive web app for dynamic forecasting and trend analysis.
