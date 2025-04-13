import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Read CSV file with stock data
df = pd.read_csv('stocks.csv')

# Define the date ranges for returns
date_ranges = {
    '1M': datetime.now() - timedelta(days=30),
    '3M': datetime.now() - timedelta(days=90),
    '6M': datetime.now() - timedelta(days=180),
    '1Y': datetime.now() - timedelta(days=365)
}


# Function to get stock data using yfinance
def get_stock_data(symbol):
    try:
        stock_data = yf.download(symbol + '.NS', period='1y', interval='1d')  # 1 year of daily data
        if stock_data.empty:
            print(f"❌ No data found for {symbol}")
            return None
        return stock_data
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None


# Processing the stocks and calculating returns
results = []

print("\n📈 Fetching stock data and calculating returns...")

for symbol in df['Symbol']:
    print(f"📊 Processing {symbol}...")
    stock_data = get_stock_data(symbol)

    if stock_data is None:
        continue  # Skip if no data found

    # Ensure stock_data index is timezone-naive (remove timezone info)
    stock_data.index = stock_data.index.tz_localize(None)

    # Get the current price (latest closing price)
    current_price = stock_data['Close'].iloc[-1]

    # Calculate returns for different time frames based on historical data
    returns = {}
    for label, start_date in date_ranges.items():
        # Remove timezone info from start_date (make it naive)
        start_date = start_date.replace(tzinfo=None)

        # Get the close price for the specified date range (e.g., 1M, 3M, etc.)
        historical_data = stock_data.loc[stock_data.index >= start_date]
        if historical_data.empty:
            returns[label] = None  # No data found for the given range
        else:
            past_price = historical_data['Close'].iloc[0]
            returns[label] = (current_price - past_price) / past_price

    # Sharpe ratio calculation (simplified version, using daily returns)
    daily_returns = stock_data['Close'].pct_change().dropna()  # Daily returns
    if daily_returns.std().any() != 0:  # Check if the standard deviation is not zero
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan  # Assign NaN if standard deviation is zero

    # Calculate average return
    avg_return = np.nanmean(list(returns.values()))

    results.append({
        'Symbol': symbol.replace('.NS', ''),
        '1M Return (%)': round(returns.get('1M', 0) * 100, 2),
        '3M Return (%)': round(returns.get('3M', 0) * 100, 2),
        '6M Return (%)': round(returns.get('6M', 0) * 100, 2),
        '1Y Return (%)': round(returns.get('1Y', 0) * 100, 2),
        'Average Return (%)': round(avg_return * 100, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2)
    })

# === Output ===
result_df = pd.DataFrame(results)
top_20 = result_df.sort_values(by='Average Return (%)', ascending=False).head(20)

# Show in terminal
print("\n✅ Top 20 Stocks by Average Return:\n")
print(top_20.to_string(index=False))

# Save to file
top_20.to_csv('top_20_sorted_returns.csv', index=False)
print("\n📁 Results saved to 'top_20_sorted_returns.csv'")
