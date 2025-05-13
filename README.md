📈 Stock Price Prediction with LSTM (PyTorch)
A Streamlit web application that uses a deep learning LSTM model (implemented in PyTorch) to predict stock prices based on historical data fetched via Yahoo Finance (yfinance). The app supports both Indian (NSE) and US stock symbols, and includes features like historical visualization, model training visualization, actual vs. predicted plotting, and a future 30-day forecast.

🚀 Features
📥 Input any stock symbol (e.g., SBIN.NS for NSE or AAPL for US)

📉 Train an LSTM model on 2 years of historical data

📊 Visualize:

Raw stock data

Model training loss

Actual vs predicted prices

30-day future price forecast

🧠 Built using PyTorch, with real-time interactive controls via Streamlit

🌍 Compatible with Indian (NSE) and US stock exchanges

🛠 Tech Stack
Frontend/UI: Streamlit

Data Source: Yahoo Finance via yfinance

Deep Learning: PyTorch

Data Processing: NumPy, Pandas, scikit-learn

Visualization: Matplotlib, Streamlit Charts

📦 Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/1210shivanshu/Stock-Price-Prediction-with-LSTM-
cd stock-prediction-lstm
2. Create and Activate a Virtual Environment (Optional)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or install manually:

bash
Copy
Edit
pip install streamlit yfinance torch numpy pandas matplotlib scikit-learn
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
Then open the provided local URL in your browser.

📸 Screenshots
📉 Training Loss	📊 Actual vs Predicted	🔮 30-Day Forecast

(Optional: Add screenshots if deploying to GitHub)

💡 Example Stock Symbols
NSE (India): SBIN.NS, TATASTEEL.NS, RELIANCE.NS

US Stocks: AAPL, GOOGL, MSFT

📌 Notes
Uses a basic LSTM with 2 hidden layers and 200 hidden units.

Trains for 50 epochs by default on 80% of the historical data.

Model is retrained every time based on the selected stock symbol.

Good prediction accuracy despite slight variation in predicted values — especially promising for a first version!

🧠 Future Improvements
Add hyperparameter tuning (epochs, layers, etc.)

Cache model objects for repeated symbols

Model persistence and checkpointing

Deploy via Streamlit Cloud or Hugging Face Spaces

🤝 Contributing
Pull requests and forks are welcome. If you have suggestions or ideas to improve the model or app, feel free to open an issue!

📬 Contact
Created with ❤️ by Shivanshu

