import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import tempfile
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)




# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated): ", "AAPL,MSFT,GOOG")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

# Set date range    
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

st.sidebar.subheader("Change between technical and fundamental analysis")
options = ["Technical", "Fundamental"]
analysis_type = st.sidebar.radio("Select one:", options)

if analysis_type == "Technical":
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP","RSI", "MACD", "OBV"],
        default=["20-Day SMA"]
    )
    
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            stock_data[ticker] = data
            st.session_state["stock_data"] = stock_data
            st.success("Stock data loaded successfully for: " + ",".join(stock_data.keys()))
        else:
            st.warning(f"No data found for {ticker}.")
    st.info("Analyze data")
else: 
    st.info("Please fetch stock data using sidebar.")   

if st.sidebar.button("Analyze") and "stock_data" in st.session_state and st.session_state["stock_data"]:
    def analyze_ticker(ticker, data):
        # print(data["Open"][ticker])
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"][ticker],
                high=data["High"][ticker],
                low=data["Low"][ticker],
                close=data["Close"][ticker],
                name="Candlestick"
            )
        ])
        # def add_indicator(indicator):
        #     if indicator == "20-Day SMA":
        #         sma = data['Close'][ticker].rolling(window=20).mean()
        #         fig.add_trace(go.Scatter(x=data.index, y=sma, mode="lines", name="SMA (20)"))
        #     elif indicator == "20-Day EMA":
        #         ema = data['Close'][ticker].ewm(span=20).mean()
        #         fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name="EMA (20)"))
        #     elif indicator == "20-Day Bollinger Bands":
        #         sma = data['Close'][ticker].rolling(window=20).mean()
        #         std = data['Close'][ticker].rolling(window=20).std()
        #         bb_upper = sma + 2 * std
        #         bb_lower = sma - 2 * std
        #         fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
        #         fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        #     elif indicator == "VWAP":
        #         data['VWAP'] = (data['Close'][ticker] * data['Volume'][ticker]).cumsum() / data['Volume'][ticker].cumsum()
        #         fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
        def add_indicator(indicator, window=20, std_multiplier=2, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9):
            """
            Adds a technical indicator to a Plotly figure.

            Args:
                indicator: The type of indicator to add.
                window: The window size for moving averages and Bollinger Bands.
                std_multiplier: The standard deviation multiplier for Bollinger Bands.
                rsi_window: The window size for RSI calculation.
                macd_fast: The fast period for MACD.
                macd_slow: The slow period for MACD.
                macd_signal: The signal period for MACD.
            """

            if indicator not in ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "RSI", "MACD", "OBV"]:
                print(f"Error: Invalid indicator '{indicator}'")
                return

            if 'Close' not in data or ('Volume' not in data and indicator in ['VWAP', 'OBV']):
                print(f"Error: Missing required columns in data for ticker '{ticker}'")
                return

            close_prices = data['Close'][ticker]
            index = data.index

            match indicator:
                case "20-Day SMA":
                    sma = close_prices.rolling(window=window).mean()
                    fig.add_trace(go.Scatter(x=index, y=sma, mode="lines", name=f"SMA ({window})"))
                case "20-Day EMA":
                    ema = close_prices.ewm(span=window).mean()
                    fig.add_trace(go.Scatter(x=index, y=ema, mode='lines', name=f"EMA ({window})"))
                case "20-Day Bollinger Bands":
                    sma = close_prices.rolling(window=window).mean()
                    std = close_prices.rolling(window=window).std()
                    bb_upper = sma + std_multiplier * std
                    bb_lower = sma - std_multiplier * std
                    fig.add_trace(go.Scatter(x=index, y=bb_upper, mode='lines', name=f'BB Upper ({window})'))
                    fig.add_trace(go.Scatter(x=index, y=bb_lower, mode='lines', name=f'BB Lower ({window})'))
                case "VWAP":
                    vwap = (close_prices * data['Volume'][ticker]).cumsum() / data['Volume'][ticker].cumsum()
                    fig.add_trace(go.Scatter(x=index, y=vwap, mode='lines', name='VWAP'))
                case "RSI":
                    rsi = calculate_rsi(close_prices, window=rsi_window)
                    fig.add_trace(go.Scatter(x=index, y=rsi, mode='lines', name=f'RSI ({rsi_window})'))
                case "MACD":
                    macd, signal = calculate_macd(close_prices, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
                    fig.add_trace(go.Scatter(x=index, y=macd, mode='lines', name=f'MACD ({macd_fast}, {macd_slow}, {macd_signal})'))
                    fig.add_trace(go.Scatter(x=index, y=signal, mode='lines', name=f'Signal ({macd_fast}, {macd_slow}, {macd_signal})'))
                case "OBV":
                    obv = calculate_obv(close_prices, data['Volume'][ticker])
                    fig.add_trace(go.Scatter(x=index, y=obv, mode='lines', name='OBV'))
        def add_fundamental_data(ticker, start_date=start_date, end_date=end_date, indicators=["Revenue", "Net Income", "EPS"]):
            """
            Fetches fundamental data from yfinance and plots specified indicators.

            Args:
                ticker (str): The stock ticker symbol (e.g., "AAPL").
                start_date (str): Start date for data retrieval (YYYY-MM-DD).
                end_date (str): End date for data retrieval (YYYY-MM-DD).
                indicators (list): List of fundamental indicators to plot.
            """

            try:
                # Fetch fundamental data from yfinance
                stock = yf.Ticker(ticker)
                income_statement = stock.income_stmt
                if income_statement.empty:
                    print(f"Error: No income statement data found for {ticker}.")
                    return

                # Prepare data for plotting
                dates = income_statement.columns

                fig = go.Figure()

                for indicator in indicators:
                    if indicator in income_statement.index:
                        values = income_statement.loc[indicator].values
                        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name=indicator))
                    elif indicator == "EPS": #special case for EPS
                        if "BasicEPS" in income_statement.index:
                            values = income_statement.loc["BasicEPS"].values
                            fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name="EPS (Basic)"))
                        elif "DilutedEPS" in income_statement.index:
                            values = income_statement.loc["DilutedEPS"].values
                            fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name="EPS (Diluted)"))
                        else:
                            print(f"Warning: EPS data not found for {ticker}.")
                    else:
                        print(f"Warning: Indicator '{indicator}' not found in income statement.")

                fig.update_layout(title=f"Fundamental Data for {ticker}", xaxis_title="Date", yaxis_title="Value")

            except Exception as e:
                print(f"An error occurred: {e}") 

            return fig
           
        if analysis_type == "Technical":                
            for ind in indicators:
                add_indicator(ind)
        else:
            fig = add_fundamental_data(ticker)

        fig.update_layout(xaxis_rangeslider_visible=False)
        # fig.update_layout(
        #     title=f"{ticker} Stock Analysis",
        #     xaxis_title="Date",
        #     yaxis_title="Price",
        #     xaxis_rangeslider_visible=False,
        #     height=600,
        #     width=1000
        # )
        # fig.show() # Only for debugging purpose

        # Save chart as a temporary PNG file and read image bytes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name
        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tmpfile_path)

        # Create an image part
        image_part = {
            "data": image_bytes,
            "mime_type": "image/png"
        }

        analysis_prompt = (
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution."
            f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators."
            f"Provide a detailed justification of your analysis, explaining what patterns, signals and trends you observe. "
            f"Then, based solely on the chart, provide a recommendation from the following options: "
            f"'Strong buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."

        )

        contents = [
            {"role": "user", "parts": [analysis_prompt]},
            {"role": "user", "parts": [image_part]}
        ]

        response = gen_model.generate_content(
            contents=contents
        )

        try: 
            result_text = response.text
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else: 
                raise ValueError("No valid JSON object found  in the response")
            
        except json.JSONDecodeError as e:
            result = {"action": "Error", "justification": f"JSON Parsing error: {e}. Raw response text: {response.text}"}
        except ValueError as ve:
            result = {"action": "Error", "justification": f"Value Error: {ve}. Raw response text: {response.text}"}
        except Exception as e:
            result = {"action": "Error", "justification": f"General Error: {e}. Raw response text: {response.text}"}
        
        return fig, result
    
        tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
        tabs = st.tabs(tab_names)

        overall_results = []

        for i, ticker in enumerate(st.session_state["stock_data"]):
            data = st.session_state["stock_data"][ticker]
            # print(data)
            fig, result = analyze_ticker(ticker, data)
            overall_results.append({"Stock": ticker, "Recomendation": result.get("action", "N/A")})
            with tabs[i+1]:
                st.subheader(f"Analysis for {ticker}")
                st.plotly_chart(fig)
                st.write("**Detailed justification:**")
                st.write(result.get("justification", "No justification provided."))

        with tabs[0]:
            st.subheader("Overall Structured Recomendations")
            df_summary = pd.DataFrame(overall_results)
            st.table(df_summary)

    def calculate_rsi(series, window=14):
        """Calculates the Relative Strength Index (RSI)."""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=window).mean()
        ma_down = down.rolling(window=window).mean()
        rsi = ma_up / ma_down
        rsi = 100 * (1 - (1 / (1 + rsi)))
        return rsi

    def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        ema_fast = series.ewm(span=fast_period).mean()
        ema_slow = series.ewm(span=slow_period).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        return macd, signal
    
    def calculate_obv(close, volume):
        """Calculates the On-Balance Volume (OBV)."""
        obv = (volume * (close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0))).cumsum()
        return obv
    
    
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    overall_results = []

    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        # print(data)
        fig, result = analyze_ticker(ticker, data)
        overall_results.append({"Stock": ticker, "Recomendation": result.get("action", "N/A")})
        with tabs[i+1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig)
            st.write("**Detailed justification:**")
            st.write(result.get("justification", "No justification provided."))

    with tabs[0]:
        st.subheader("Overall Structured Recomendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)

