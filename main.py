import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

if key is None:
    st.error("API key not found! Please set the 'OPENAI_API_KEY' environment variable.")
else:
    client = OpenAI(api_key=key)
    st.success("API key loaded successfully!")

# Create a function to fetch Historical Stock data
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker and date range.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.warning("No data found for the given date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Define a Trading Strategy Class Using OOP
class TradingStrategy:
    """
    Base class for trading strategies.
    """
    def generate_signals(self, data):
        raise NotImplementedError("Should implement generate_signals() method")

# Implement a Simple Moving Average Strategy
class MovingAverageStrategy(TradingStrategy):
    """
    Implements a simple moving average crossover strategy.
    """
    def __init__(self, short_window=40, long_window=100):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # Create short and long simple moving averages
        signals['short_mavg'] = data['Close'].rolling(
            window=self.short_window, min_periods=1, center=False
        ).mean()
        signals['long_mavg'] = data['Close'].rolling(
            window=self.long_window, min_periods=1, center=False
        ).mean()

        # Generate signals
        signals['signal'][self.short_window:] = np.where(
            signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1.0, 0.0
        )

        # Generate trading orders
        signals['positions'] = signals['signal'].diff()

        return signals

# Backtest the trading strategy
def backtest_strategy(data, signals, initial_capital=100000.0):
    """
    Backtests the trading strategy.
    """
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Position'] = signals['signal'] * 100

    portfolio = positions.multiply(data['Adj Close'], axis=0)
    pos_diff = positions.diff()

    # Calculate portfolio holdings
    portfolio['holdings'] = positions['Position'] * data['Adj Close']
    # Calculate Cash
    portfolio['cash'] = initial_capital - (pos_diff['Position'] * data['Adj Close']).cumsum()
    # Calculate total assets
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    # Calculate returns
    portfolio['returns'] = portfolio['total'].pct_change()

    return portfolio

# Use Generative AI to Optimize Strategy Parameters
def optimize_strategy_with_ai(ticker):
    """
    Uses OpenAI's GPT to suggest optimal strategy parameters.
    """
    prompt = f"Suggest optimal short_window and long_window periods for a moving average crossover strategy for {ticker} stock."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial expert specialized in algorithmic trading."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        suggestion = response.choices[0].message.content.strip()
        
        # Estimate API cost (assuming $0.002 per 1K tokens for gpt-3.5-turbo)
        tokens_used = response.usage.total_tokens
        estimated_cost = (tokens_used / 1000) * 0.002
        st.info(f"Estimated API cost for this request: ${estimated_cost:.4f}")
        
        return suggestion
    except Exception as e:
        st.error(f"Error with OpenAI: {e}")
        return None

# Build the Streamlit App Interface
# Streamlit App title
st.title("AI-Powered Algorithmic Trading Strategy Backtester")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")
start_date = st.date_input("Start Date:")
end_date = st.date_input("End Date:")
initial_capital = st.number_input("Initial Capital ($):", value=100000.0)

data = None

# Fetch and display data
if st.button("Fetch Data"):
    data = fetch_stock_data(ticker, start_date, end_date)
    if data is not None:
        st.subheader("Stock Data")
        st.write(data.tail())
        st.line_chart(data['Close'])

# Integrate Strategy, Backtesting, and AI Optimization
if 'data' in locals() and data is not None:
    # Get AI suggestions
    st.subheader("AI Optimization Suggestions")
    suggestion = optimize_strategy_with_ai(ticker)
    if suggestion:
        st.write(f"AI Suggestion: {suggestion}")

    # User inputs for strategy parameters
    short_window = st.number_input("Short Moving Average Window:", value=40, min_value=1, step=1)
    long_window = st.number_input("Long Moving Average Window:", value=100, min_value=1, step=1)

    if st.button("Run Backtest"):
        # Initialize and run strategy
        strategy = MovingAverageStrategy(short_window=short_window, long_window=long_window)
        signals = strategy.generate_signals(data)

        # Backtest strategy
        portfolio = backtest_strategy(data, signals, initial_capital)

        # Display results
        st.subheader("Portfolio Performance")
        st.line_chart(portfolio['total'])
        st.subheader("Strategy Signals")
        st.write(signals.tail())

        # Option to save results
        if st.button("Save Results"):
            try:
                # Create directory if it doesn't exist
                if not os.path.exists('results'):
                    os.makedirs('results')

                # Save portfolio performance to CSV
                portfolio.to_csv(f'results/{ticker}_portfolio.csv')
                # Save signals performance to CSV
                signals.to_csv(f'results/{ticker}_signals.csv')

                st.success("Results saved successfully!")
            except Exception as e:
                st.error(f"Error saving results: {e}")
else:
    st.info("Please fetch data to proceed.")