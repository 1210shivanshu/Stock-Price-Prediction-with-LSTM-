import streamlit as st
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Title
st.title("📈 Stock Price Prediction with LSTM (PyTorch)")
st.markdown("Predict future stock prices using an LSTM-based deep learning model.")

# Input
symbol = st.text_input("Enter Stock Symbol (e.g., SBIN.NS)", "SBIN.NS")
time_step = 60
epochs = 50

# Data loading
@st.cache_data
def load_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

today = datetime.today()
start_date = (today - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')
df = load_data(symbol, start_date, end_date)

if df.empty:
    st.error("Failed to fetch data. Please check the symbol.")
    st.stop()

st.subheader(f"🔍 Historical Stock Prices ({df.index.min().date()} to {df.index.max().date()})")

# Ensure datetime index
df.index = pd.to_datetime(df.index)

st.write("First few rows of data:")
st.write(df[['Close']].head())

st.write("Missing values in Close column:", df['Close'].isna().sum())

# Display the line chart (Close prices)
#st.line_chart(df['Close'])

st.markdown("### 📊 Statistical Summary")
st.write(df.describe())


# Preprocess
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=200, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
train_losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train.unsqueeze(-1))
    loss = criterion(output.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

st.subheader("📉 Training Loss")
st.line_chart(train_losses)

# Test
model.eval()
with torch.no_grad():
    predictions = model(X_test.unsqueeze(-1)).squeeze()
    test_loss = criterion(predictions, y_test).item()
st.metric("🧪 Test Loss (MSE)", f"{test_loss:.5f}")

# Actual vs Predicted
full_actual = scaler.inverse_transform(scaled_data)
full_predicted = np.empty_like(full_actual)
full_predicted[:] = np.nan
test_start_index = train_size + time_step
full_predicted[test_start_index:test_start_index + len(predictions)] = scaler.inverse_transform(
    predictions.detach().numpy().reshape(-1, 1))

st.subheader("📊 Actual vs Predicted Prices")
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df.index, full_actual, label="Actual", color="black")
ax1.plot(df.index, full_predicted, label="Predicted", color="red")
ax1.axvline(df.index[train_size], color="blue", linestyle="--", label="Train/Test Split")
ax1.set_title(f"{symbol} Stock Price Prediction")
ax1.legend()
st.pyplot(fig1)

# Future prediction
st.subheader("🔮 Future 30-Day Forecast")
last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)
future_predictions = []

with torch.no_grad():
    for _ in range(30):
        input_tensor = torch.tensor(last_60_days).float()
        predicted = model(input_tensor)
        future_predictions.append(predicted.item())
        last_60_days = np.roll(last_60_days, -1)
        last_60_days[0, -1, 0] = predicted.item()

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=today + timedelta(days=1), periods=30)

fig2, ax2 = plt.subplots()
ax2.plot(future_dates, future_predictions, color="orange", label="Predicted")
ax2.set_title("Next 30 Days Stock Price Prediction")
ax2.legend()
st.pyplot(fig2)
