import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yfinance as yf
import requests
import plotly.graph_objects as go
import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta
import threading
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="Stock Market Prediction", page_icon="📈")

#define the CSS styles**
custom_css = """
    <style>
    body { transition: all 0.3s ease-in-out; }
    .stButton>button {
        transition: 0.3s ease-in-out;
        background-color: #1E88E5 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stButton>button:hover {
        background-color: #1565C0 !important;
        transform: scale(1.05);
    }
    .stMarkdown {
        transition: 0.3s ease-in-out;
    }
    .stPlotlyChart {
        transition: transform 0.3s ease-in-out;
    }
    .stPlotlyChart:hover {
        transform: scale(1.02);
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load LSTM Model
@st.cache_resource
def load_lstm_model():
    try:
        model = tf.keras.models.load_model("lstm_model.h5")  # Ensure correct model path
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

lstm_model = load_lstm_model()

# Load Stock Data from Default Location
@st.cache_data
def load_stock_data():
    try:
        default_path = "AAPL.csv"  # Looks for this file in the script’s directory
        if os.path.exists(default_path):
            df = pd.read_csv(default_path, parse_dates=["Date"])
            return df
        else:
            st.error(f"❌ Error: 'stock_data.csv' not found. Please add it to the script directory.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading stock data: {str(e)}")
        return None

df_stock = load_stock_data()


# Function to get Real-time Stock Price
def get_real_time_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="1d")
        if stock_data.empty:
            return "Stock data not available"
        return round(stock_data["Close"].iloc[-1], 2)
    except:
        return "Error fetching stock price"

# Function to Predict Future Stock Prices (Using Slider Only)
def predict_stock_prices(start_date, forecast_days):
    try:
        if lstm_model is None:
            return None, "❌ Model not loaded. Please check."

        if df_stock is None or df_stock.empty:
            return None, "⚠️ No stock data available."

        # Convert Dates & Set Index
        df_stock["Date"] = pd.to_datetime(df_stock["Date"], errors="coerce")
        df_stock.set_index("Date", inplace=True)

        # Use last 60 days of stock data for prediction
        df = df_stock.tail(60).copy()
        if df.empty:
            return None, f"⚠️ No valid stock data available for prediction."

        # Prepare Data for Prediction
        df_close = df[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_close)

        # Convert last 60 days into LSTM-friendly shape
        last_60_days = scaled_data.reshape(1, 60, 1)

        # Generate Predictions
        predictions = []
        for _ in range(forecast_days):
            pred = lstm_model.predict(last_60_days, verbose=0)
            predictions.append(pred[0, 0])
            last_60_days = np.roll(last_60_days, -1, axis=1)  # Shift left
            last_60_days[0, -1, 0] = pred

        # Convert predictions back to original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Generate Correct Forecast Dates
        forecast_dates = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_days)

        # Create DataFrame
        prediction_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': predictions.flatten()})

        return prediction_df, None

    except Exception as e:
        return None, f"❌ Error generating predictions: {str(e)}"
    

# Function to Calculate Moving Averages Using Model Predictions
def calculate_moving_averages(df, window_sizes=[10, 20, 50]):
    moving_averages = {}
    for window in window_sizes:
        moving_averages[f"SMA_{window}"] = df['Close'].rolling(window=window).mean()  # Simple Moving Average
        moving_averages[f"EMA_{window}"] = df['Close'].ewm(span=window, adjust=False).mean()  # Exponential Moving Average
    return pd.DataFrame(moving_averages, index=df.index)

# Cache FinBERT Model to Prevent Reloading Every Time
@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

finbert_sentiment = load_finbert()

# Function to Analyze Sentiment Using FinBERT (Batch Processing for Speed)
def analyze_sentiment_finbert_batch(texts):
    results = finbert_sentiment(texts)
    sentiment_labels = []
    for result in results:
        label = result["label"]
        if label == "positive":
            sentiment_labels.append("📈 Positive")
        elif label == "negative":
            sentiment_labels.append("📉 Negative")
        else:
            sentiment_labels.append("⚖️ Neutral")
    return sentiment_labels

# Function to fetch Live Stock News 
def fetch_stock_news(symbol, max_articles=15):
    try:
        url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
        feed = feedparser.parse(url)

        if not feed.entries:
            return ["⚠️ No recent news available."]

        news_titles = [entry.title for entry in feed.entries[:max_articles]]
        sentiment_labels = analyze_sentiment_finbert_batch(news_titles)  

        news_items = [f"[{entry.title}]({entry.link}) - {sentiment}"
                      for entry, sentiment in zip(feed.entries[:max_articles], sentiment_labels)]

        return news_items

    except Exception as e:
        return [f"❌ Error fetching news: {str(e)}"]
    
# Function to get Stock Volatility
def get_stock_volatility(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo")
    daily_changes = hist["Close"].pct_change().dropna()
    volatility = np.std(daily_changes) * 100  # Convert to percentage
    return round(volatility, 2)

# Function to get 52-Week High/Low and Market Cap
def get_stock_summary(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    return {
        "52_Week_High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52_Week_Low": info.get("fiftyTwoWeekLow", "N/A"),
        "Market_Cap": info.get("marketCap", "N/A")
    }

# Function to Fetch Real-Time Stock Prices for Ticker
def fetch_live_stock_prices(symbols=["AAPL", "TSLA", "AMZN", "GOOG", "MSFT", "NFLX", "NVDA", "META"]):
    try:
        tickers = yf.Tickers(" ".join(symbols))
        stock_prices = {symbol: round(tickers.tickers[symbol].history(period="1d")["Close"].iloc[-1], 2) for symbol in symbols}
        return stock_prices
    except:
        return {}

# Function to Create Scrolling Ticker Text
def generate_ticker_text(stock_prices):
    ticker_text = "  |  ".join([f"{symbol}: ${price}" for symbol, price in stock_prices.items()])
    return f"📈 {ticker_text} 📉"

# Function to Fetch Live Market Index Performance Using Yahoo Finance
def fetch_market_indices():
    indices = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI"
    }

    market_data = {}

    try:
        for name, symbol in indices.items():
            index = yf.Ticker(symbol)
            hist = index.history(period="1d")

            if hist.empty:
                return {"Error": "No data available for market indices"}

            current_price = round(hist["Close"].iloc[-1], 2)
            previous_close = round(hist["Open"].iloc[-1], 2)
            change = round(current_price - previous_close, 2)
            change_percentage = round((change / previous_close) * 100, 2)

            market_data[name] = {
                "Current": current_price,
                "Change": change,
                "Change %": change_percentage
            }

        return market_data
    except Exception as e:
        return {"Error": str(e)}
    
# Function to Calculate Forecast Days Based on Selected Date Range
def calculate_forecast_days(start_date, end_date):
    return (end_date - start_date).days

def get_dynamic_threshold(stock_symbol, period="30d"):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period=period)

    if hist.empty:
        return 2  # Default fallback threshold

    hist["Daily Change %"] = hist["Close"].pct_change() * 100
    dynamic_threshold = np.abs(hist["Daily Change %"]).mean()
    return round(dynamic_threshold, 2)

def generate_ai_recommendation(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="7d")  # Get last 7 days of data

        if hist.empty:
            return "⚠️ Not Enough Data"

        # ✅ Calculate % Change in Price
        recent_close = hist["Close"].iloc[-1]
        previous_close = hist["Close"].iloc[-2]
        price_change = ((recent_close - previous_close) / previous_close) * 100

        # ✅ AI-Based Recommendation Logic
        if price_change > 2:
            return f"📈 **BUY** (Price up {price_change:.2f}% - Strong Momentum)"
        elif price_change < -2:
            return f"📉 **SELL** (Price down {price_change:.2f}% - Downtrend)"
        else:
            return f"⚖️ **HOLD** (Price changed {price_change:.2f}% - Stable Trend)"

    except Exception as e:
        return f"❌ Error: {str(e)}"



    
@st.cache_data
def generate_forecast_chart(dates, prices):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines+markers", name="Predicted Prices"))
    fig.update_layout(title="Stock Price Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig


# Main Page Title
st.title("📈 Stock Market Prediction")
st.markdown("### AI-powered predictions for stock prices")

# Fetch Stock Prices
stock_prices = fetch_live_stock_prices()

# Create a Scrolling Marquee
if stock_prices:
    ticker_text = generate_ticker_text(stock_prices)
    st.markdown(
    f"""
    <marquee behavior="scroll" direction="left" scrollamount="5" style="font-size: 20px; font-weight: bold; color: #E91E63;">
    {ticker_text}
    </marquee>
    """,
    unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Unable to fetch live stock prices. Please check your internet connection or API limits.")


# Main Layout: Left Sidebar + Center Content + Right Sidebar
left_sidebar, main_content, right_sidebar = st.columns([1, 2, 1])

# LEFT SIDEBAR: STOCK SELECTION
with left_sidebar:
    st.markdown("### 🔍 Stock Selection")

    # Fetch stock list
    @st.cache_data
    def get_stock_list():
        try:
            tickers = yf.Tickers("AAPL TSLA AMZN GOOG MSFT NFLX NVDA META")
            return [{"symbol": key, "name": tickers.tickers[key].info.get("longName", "N/A")} for key in tickers.tickers.keys()]
        except:
            return [
                {"symbol": "AAPL", "name": "Apple Inc."},
                {"symbol": "TSLA", "name": "Tesla Inc."},
                {"symbol": "AMZN", "name": "Amazon.com Inc."},
                {"symbol": "GOOG", "name": "Alphabet Inc. (Google)"},
                {"symbol": "MSFT", "name": "Microsoft Corporation"},
            ]

    stock_list = get_stock_list()
    stock_options = {stock["symbol"]: f"{stock['symbol']} - {stock['name']}" for stock in stock_list}
    stock_symbol = st.selectbox("Choose a Stock:", options=list(stock_options.keys()), format_func=lambda x: stock_options[x])
    
    # Display Real-time Price
    st.markdown("### 💰 Real-time Stock Price")
    st.info(f"**${get_real_time_price(stock_symbol)}**")

    # Divider
    st.markdown("---")
    
    # LIVE MARKET INDEX PERFORMANCE
    st.markdown("### 📊 Live Market Index Performance")
    
    # Fetch Index Data
    market_indices = fetch_market_indices()
    
    # Display Market Index Performance
    if "Error" in market_indices:
        st.warning(f"⚠️ {market_indices['Error']}")
    else:
        for index, data in market_indices.items():
            change_color = "🔴" if data["Change"] < 0 else "🟢"
            st.write(f"{index}: **${data['Current']}** ({change_color} {data['Change']}%)")

    # Divider
    st.markdown("---")
    
    # STOCK INSIGHTS
    st.markdown("### 📊 Stock Insights")
    
    # Fetch stock summary
    stock_summary = get_stock_summary(stock_symbol)

    # Stock Volatility
    st.info(f"📊 **Stock Volatility:** {get_stock_volatility(stock_symbol)}%")

    # 52-Week High/Low & Market Cap (Styled for Readability)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"📈 **52-Week High:**")
        st.success(f"${stock_summary['52_Week_High']}") 

    with col2:
        st.write(f"📉 **52-Week Low:**")
        st.error(f"${stock_summary['52_Week_Low']}") 

    # Market Cap Display
    st.write(f"💰 **Market Cap:**")
    st.warning(f"${stock_summary['Market_Cap']}")  # Yellow background for emphasis
    
    # Divider
    st.markdown("---")    

# MAIN CONTENT: STOCK DATA & ANALYSIS
with main_content:
    st.header(f"📊 Apple Stock Overview")
    
    #  **AI Stock Prediction Section**
    st.title("🔮 AI Stock Prediction")

    # 📆 **Forecast Days Slider (7-30 Days)**
    selected_forecast_days = st.slider("📆 Select Forecast Days", min_value=7, max_value=30, value=15)

    # 🔍 **Predict Button**
    if st.button("🔍 Predict Stock Prices"):
        with st.spinner("Generating AI Predictions..."):
            start_date_fixed = pd.to_datetime("2019-12-31")  # Fixed base start date
            end_date_fixed = start_date_fixed + pd.Timedelta(days=selected_forecast_days - 1)

            prediction_df, error_message = predict_stock_prices(start_date_fixed, selected_forecast_days)

            if error_message:
                st.error(error_message)
            else:
                # Display Predictions
                # st.success("✅ Prediction Complete!")
                st.write(prediction_df)

                # 📈 **Line Graph of Predictions**
                fig = px.line(prediction_df, x='Date', y='Predicted Price', title="📈 Stock Price Forecast")
                st.plotly_chart(fig, use_container_width=True)

                # 📥 **Download CSV**
                csv = prediction_df.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions", data=csv, file_name="stock_predictions.csv", mime="text/csv")

      
        # Reset Session State Variables
        st.session_state["slider_used"] = False  # Reset slider tracking
        st.session_state["use_date_range"] = False  # Default to slider input

    st.header("Moving Averages & Candlestick Chart")
    st.subheader("📉 Moving Averages")

    # Compute Moving Averages
    moving_avg_df = calculate_moving_averages(df_stock)

    # Moving Averages Chart with Custom Colors
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=moving_avg_df.index, y=moving_avg_df["SMA_10"], mode="lines", name="10-Day SMA", line=dict(color="#FF5733")))
    fig_ma.add_trace(go.Scatter(x=moving_avg_df.index, y=moving_avg_df["SMA_20"], mode="lines", name="20-Day SMA", line=dict(color="#33FF57")))
    fig_ma.add_trace(go.Scatter(x=moving_avg_df.index, y=moving_avg_df["SMA_50"], mode="lines", name="50-Day SMA", line=dict(color="#3357FF")))
    fig_ma.add_trace(go.Scatter(x=moving_avg_df.index, y=moving_avg_df["EMA_10"], mode="lines", name="10-Day EMA", line=dict(color="#FF33A8", dash="dot")))
    fig_ma.add_trace(go.Scatter(x=moving_avg_df.index, y=moving_avg_df["EMA_20"], mode="lines", name="20-Day EMA", line=dict(color="#FFD700", dash="dot")))
    fig_ma.add_trace(go.Scatter(x=moving_avg_df.index, y=moving_avg_df["EMA_50"], mode="lines", name="50-Day EMA", line=dict(color="#8A2BE2", dash="dot")))

    fig_ma.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)", plot_bgcolor="rgba(240, 240, 240, 0.9)")
    st.plotly_chart(fig_ma)

    st.subheader("🕯️ Candlestick Chart")

    stock_data = yf.Ticker(stock_symbol).history(period="1y")
    fig_candlestick = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        increasing_line_color="green",
        decreasing_line_color="red"
    )])

    fig_candlestick.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price (USD)", plot_bgcolor="rgba(240, 240, 240, 0.9)")
    st.plotly_chart(fig_candlestick)
        
    
# RIGHT SIDEBAR: NEWS & AI RECOMMENDATIONS
with right_sidebar:
    st.header("📰 Live Stock News & Sentiment")
    if "news_index" not in st.session_state:
        st.session_state.news_index = 0
        st.session_state.show_more = False  # Track button state
    
    # Show only the top 5 news headlines by default
    news_batch_size = 5  # Number of news articles to show at a time
    stock_news = fetch_stock_news(stock_symbol, max_articles=15)  # Fetch max 15 articles

    # Show initial batch of news
    news_to_display = stock_news[: st.session_state.news_index + news_batch_size]
    for news in news_to_display:
        st.markdown(news, unsafe_allow_html=True)

    # Load More or Less button
    if st.session_state.news_index + news_batch_size < len(stock_news):
        if st.button("More+" if not st.session_state.show_more else "Less-"):
            if not st.session_state.show_more:
                st.session_state.news_index += news_batch_size  # Load next set of news
                st.session_state.show_more = True
            else:
                st.session_state.news_index -= news_batch_size  # Collapse news back
                st.session_state.show_more = False
            st.rerun()  # Refresh page to show updated news

    # Add a separator for clarity
    st.markdown("---")
      
    # DISPLAY AI BUY/SELL RECOMMENDATION
    st.header("🌟 AI Buy/Sell Recommendation")
    ai_recommendation = generate_ai_recommendation(stock_symbol)
    st.info(ai_recommendation)

    # Add a separator for clarity
    st.markdown("---")

st.write("🔥 **Powered by AI, Built for Traders**")
