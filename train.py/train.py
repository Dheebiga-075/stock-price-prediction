import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Save data
data.to_csv("data/stock_data.csv")

# Use Close price
data = data[['Close']]

# Create target column (next day price)
data['Prediction'] = data['Close'].shift(-1)

# Drop last row
data = data.dropna()

# Features and labels
X = data[['Close']]
y = data['Prediction']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained and saved!")