
# AI-Based IPO Success Predictor with Market Sentiment Analysis

This project is an AI-based tool designed to predict the success of upcoming Initial Public Offerings (IPOs) by analyzing historical IPO data, financial metrics, and market sentiment. The prediction model utilizes machine learning techniques, including linear regression, random forest, and XGBoost, to assess IPO performance based on various features like market capitalization, P/E ratio, dividend yield, and sentiment scores derived from related news articles.

## Project Overview

IPOs are speculative and high-risk investments, and investors often struggle to assess their potential. This project aims to provide a predictive analysis based on both fundamental and sentiment-driven insights to classify IPOs as high-risk or high-potential. By training on historical IPO data and real-time sentiment analysis, the tool outputs a performance prediction, helping investors make informed decisions.

## Features

- **IPO Data Retrieval**: Fetches market data from Yahoo Finance, including market capitalization, P/E ratio, and dividend yield.
- **Sentiment Analysis**: Analyzes news articles related to the company’s ticker using the News API and TextBlob for sentiment scoring.
- **Machine Learning Models**: Trains multiple models (Linear Regression, Random Forest, and XGBoost) and selects the best-performing model based on R² score.
- **Model Serialization**: Saves the best-performing model using pickle for future predictions on new IPO data.

## Getting Started

### Prerequisites

- Python 3.7 or later
- Libraries: Install the required libraries via pip:
  ```bash
  pip install yfinance requests textblob pandas scikit-learn xgboost
  ```
- **News API Key**: Sign up on [NewsAPI](https://newsapi.org/) to get an API key for fetching news articles.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ipo-success-predictor.git
   cd ipo-success-predictor
   ```
2. Install the required Python packages as mentioned in the Prerequisites.

3. Replace `news_api_key` in the code with your News API key.

## Usage

1. **Data Collection**: Add the ticker symbols for the companies you want to analyze in the `tickers` list.

2. **Run the Code**: Execute the script to collect IPO data, perform sentiment analysis, and train the models.

   ```bash
   python ipo_predictor.py
   ```

3. **View Results**: The script will display the Mean Absolute Error (MAE) and R² score for each model, select the best one, and save it as `best_ipo_predictor_model.pkl`.

4. **Predict New IPO Performance**: Load the saved model to predict the IPO performance on new data.

## Code Structure

- **get_ipo_data(ticker)**: Fetches market data for a specified ticker from Yahoo Finance.
- **fetch_news(company_name, api_key)**: Retrieves related news articles using News API.
- **analyze_sentiment(articles)**: Calculates a sentiment score based on fetched news articles.
- **Model Training and Evaluation**: Trains Linear Regression, Random Forest, and XGBoost models to predict IPO performance.
- **Serialization**: Saves the best model for later use in making new predictions.

## Example

Below is an example of how to load the saved model and make a prediction:

```python
import pickle
import pandas as pd

# Load the best saved model
with open("best_ipo_predictor_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Prepare sample data for prediction
new_data = pd.DataFrame({
    "Market_Cap": [1e9],
    "P/E_Ratio": [60],
    "Dividend_Yield": [0.01],
    "Sentiment_Score": [0.5]
})

# Make a prediction
new_prediction = loaded_model.predict(new_data)
print("Predicted IPO Performance:", new_prediction[0])
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Yahoo Finance API](https://finance.yahoo.com/)
- [NewsAPI](https://newsapi.org/)
- [TextBlob for Sentiment Analysis](https://textblob.readthedocs.io/en/dev/)

