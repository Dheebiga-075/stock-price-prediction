import yfinance as yf
import matplotlib.pyplot as plt

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Print first few rows
print(data.head())

# Plot closing price
data['Close'].plot(title="Stock Price")
plt.show()