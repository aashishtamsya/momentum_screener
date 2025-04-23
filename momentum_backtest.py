import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Parameters
START_DATE = '2023-01-01'
END_DATE = '2024-01-01'
INITIAL_CAPITAL = 1_000_000
TOP_N = 20
STOP_LOSS_PCT = -0.20
MOMENTUM_LOOKBACK_DAYS = 63  # Approx 3 months of trading days
MIN_UNIVERSE_SIZE = 100  # Fallback in case of missing data

# Load NIFTY 500 tickers (this is a placeholder; replace with actual list if available)
nifty_500 = pd.read_csv('ind_nifty500list.csv')['Symbol'].tolist()
nifty_500 = [ticker + ".NS" for ticker in nifty_500]  # Yahoo Finance format

# Download historical data
data = yf.download(nifty_500, start=START_DATE, end=END_DATE)['Adj Close']
data = data.dropna(axis=1, thresh=0.9 * len(data))  # Drop stocks with too much missing data

# Fill missing data
prices = data.fillna(method='ffill')

# Monthly rebalancing dates
rebalance_dates = prices.resample('M').first().index

portfolio = pd.DataFrame(index=prices.index, columns=['Equity'])
portfolio['Equity'].iloc[0] = INITIAL_CAPITAL
holdings = {}

for date in rebalance_dates:
    if date not in prices.index:
        continue

    current_prices = prices.loc[date]
    recent_returns = (prices.loc[date] / prices.loc[date - timedelta(days=MOMENTUM_LOOKBACK_DAYS)] - 1)
    ranked = recent_returns.dropna().sort_values(ascending=False)
    top_momentum = ranked.head(TOP_N).index.tolist()
    top_50 = ranked.head(50).index.tolist()

    # Check exits
    for ticker in list(holdings.keys()):
        entry_price = holdings[ticker]['entry_price']
        current_price = current_prices.get(ticker, np.nan)

        if np.isnan(current_price):
            continue

        ret = (current_price - entry_price) / entry_price
        if ret <= STOP_LOSS_PCT or ticker not in top_50:
            del holdings[ticker]

    # Add new entries
    new_picks = [ticker for ticker in top_momentum if ticker not in holdings]
    cash = portfolio['Equity'].loc[date]
    num_positions = len(holdings) + len(new_picks)

    if num_positions > 0:
        allocation_per_stock = cash / num_positions

    for ticker in new_picks:
        price = current_prices.get(ticker, np.nan)
        if not np.isnan(price):
            holdings[ticker] = {'entry_price': price, 'shares': allocation_per_stock / price}

    # Update portfolio value daily until next rebalance
    next_date_idx = prices.index.get_loc(date)
    next_rebalance = prices.index[next_date_idx + 1] if next_date_idx + 1 < len(prices.index) else prices.index[-1]
    date_range = prices.loc[date:next_rebalance].index

    for d in date_range:
        daily_value = 0
        for ticker, pos in holdings.items():
            price = prices.get(ticker, pd.Series()).get(d, np.nan)
            if not np.isnan(price):
                daily_value += pos['shares'] * price
        portfolio.loc[d, 'Equity'] = daily_value

# Forward-fill equity values
portfolio['Equity'] = portfolio['Equity'].ffill()

# Plot equity curve
portfolio['Equity'].plot(figsize=(12, 6), title='Price Momentum Strategy (Top 20)', ylabel='Portfolio Value')
plt.grid()
plt.show()
