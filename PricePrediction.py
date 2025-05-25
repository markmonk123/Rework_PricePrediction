```python
import asyncio
import websockets
import pandas as pd
import numpy as np
from cdp.client import Client as CDPClient  # Coinbase SDK
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json

# ---------------------- Configuration ----------------------

API_KEY = "your_api_key"  # Replace with your API key
API_SECRET = "your_api_secret"  # Replace with your API secret
NODE_SERVER_URL = "ws://localhost:8080"  # WebSocket server URL
coinbase_client = CDPClient(api_key=API_KEY, api_secret=API_SECRET)  # Initialize Coinbase client


# ---------------------- Indicators ----------------------

def calculate_dma(df):
    """Calculate Directional Moving Average (DMA)."""
    df["DMA"] = (df["high"] + df["low"] + df["close"]) / 3
    return df


def calculate_linear_regression_slope(df, period=20):
    """Calculate the linear regression slope."""
    prices = np.array(df["close"])
    x = np.arange(len(prices)).reshape(-1, 1)
    slopes = np.zeros(len(df))

    for i in range(period - 1, len(df)):
        x_subset = x[i - period + 1 : i + 1]
        y_subset = prices[i - period + 1 : i + 1]
        lr_model = LinearRegression().fit(x_subset, y_subset)
        slopes[i] = lr_model.coef_  # Slope of the line

    df["Slope"] = slopes
    return df


def calculate_dpo(df, period=20):
    """Calculate the Detrended Price Oscillator (DPO)."""
    shifted_prices = df["close"].shift(period // 2)
    sma = df["close"].rolling(window=period).mean()
    df["DPO"] = shifted_prices - sma
    return df


def calculate_indicators(df):
    """Calculate and add all indicators (DMA, Slope, DPO)."""
    df = calculate_dma(df)
    df = calculate_linear_regression_slope(df)
    df = calculate_dpo(df)
    return df


# ---------------------- Market Position Logic ----------------------

def determine_market_position(df):
    """Determine if it's a 'Buy', 'Sell', or 'Hold'."""
    if df.empty or len(df) == 0:
        return 'Hold'  # Default to Hold for empty or invalid data

    latest = df.iloc[-1]
    slope = latest["Slope"]
    dpo = latest["DPO"]
    close = latest["close"]
    dma = latest["DMA"]

    if slope > 0 and dpo > 0 and close > dma:
        return "Buy"  # Bullish market
    elif slope < 0 and dpo < 0 and close < dma:
        return "Sell"  # Bearish market
    else:
        return "Hold"  # Neutral market


# ---------------------- Data Normalization ----------------------

def normalize_data(data, min_val, max_val):
    """Normalize data using Min-Max scaling."""
    return (data - min_val) / (max_val - min_val)


def denormalize_data(data, min_val, max_val):
    """Denormalize data back to original values."""
    return data * (max_val - min_val) + min_val


# ---------------------- Data Fetching ----------------------

def fetch_historical_data(asset, interval):
    """Fetch historical data using Coinbase SDK."""
    try:
        response = coinbase_client.candles.get_candles(asset, interval)
        df = pd.DataFrame(response["candles"], columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")  # Convert time to datetime
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def fetch_real_time_price(asset):
    """Fetch the latest real-time price using Coinbase SDK."""
    try:
        response = coinbase_client.market.ticker(asset)
        return float(response["price"])
    except Exception as e:
        print(f"Error fetching real-time price: {e}")
        return None


# ---------------------- LSTM Model ----------------------

class LSTM(nn.Module):
    """Define LSTM Model."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])  # Last timestep output


def train_model(df):
    """Train LSTM Model."""
    training_data = df["close"].values
    min_val = training_data.min()
    max_val = training_data.max()
    normalized_data = normalize_data(training_data, min_val, max_val)

    sequence_length = 10
    X, y = [], []
    for i in range(len(normalized_data) - sequence_length):
        X.append(normalized_data[i:i + sequence_length])
        y.append(normalized_data[i + sequence_length])

    X = torch.tensor(X).float().unsqueeze(-1)  # Add input size dimension
    y = torch.tensor(y).float()

    model = LSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 30
    for epoch in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model, min_val, max_val


# ---------------------- WebSocket Communication ----------------------

async def send_data():
    """WebSocket loop to send data to the server."""
    asset = "BTC-USD"  # Trading pair
    intervals = ["5m", "15m", "30m", "1h", "6h"]  # Query intervals

    async with websockets.connect(NODE_SERVER_URL) as websocket:
        print("Connecting to WebSocket server...")

        # Initial Training
        historical_data = pd.concat([fetch_historical_data(asset, interval) for interval in intervals])
        historical_data = calculate_indicators(historical_data)
        model, min_val, max_val = train_model(historical_data)

        # Continuous loop
        while True:
            current_time = datetime.utcnow()

            # Retrain every 15 minutes
            if current_time.minute % 15 == 0:
                print("Retraining the model...")
                historical_data = pd.concat([fetch_historical_data(asset, interval) for interval in intervals])
                historical_data = calculate_indicators(historical_data)
                model, min_val, max_val = train_model(historical_data)

            # Fetch real-time data
            real_time_price = fetch_real_time_price(asset)
            if real_time_price is not None:
                normalized_real_time = normalize_data(np.array([real_time_price]), min_val, max_val)[0]

                # Predict future price
                last_sequence = torch.tensor([normalized_real_time]).float().unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    normalized_future_price = model(last_sequence).item()
                future_price = denormalize_data(normalized_future_price, min_val, max_val)

                # Determine Market Position
                market_position = determine_market_position(historical_data)

                # Send WebSocket message
                message = {
                    "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "latest_price": real_time_price,
                    "future_price": future_price,
                    "recommendation": market_position,
                }
                print(f"Sending: {message}")
                await websocket.send(json.dumps(message))

            await asyncio.sleep(5)  # Update every 5 seconds


# Run the WebSocket Client
if __name__ == "__main__":
    asyncio.run(send_data())
```