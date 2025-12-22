
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

// stochastic oscillator

lowest_low = df['low'].rolling(window=30).min()
highest_high = df['high'].rolling(window=30).max()

raw_K = (df['close'] - lowest_low) / (highest_high - lowest_low) * 100

df['%K'] = raw_K.rolling(window=5).mean()
df['%D'] = df['%K'].rolling(window=3).mean()

//accumulation and distribution line indicator
mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)

df['AD'] = (mfm * df['volume']).cumsum()
df['AD_slope'] = df['AD'].diff(5)

//VIX volume indicator
df['Low_Vol'] = df['VIX'] < 20

btc = yf.download("BTC-USD", start="2022-01-01", end="2025-01-01", interval="1d")
vix = yf.download("^VIX", start="2022-01-01", end="2025-01-01", interval="1d")

df = btc.join(vix['Close'].rename('VIX'), how='inner')
df.dropna(inplace=True)

// buy/sell logic
df['Buy'] = (
    df['Low_Vol'] &
    (df['AD_slope'] > 0) &
    (df['%K'].shift(1) < df['%D'].shift(1)) &
    (df['%K'] > df['%D']) &
    (df['%K'] < 30)
)
df['Sell'] = (
    (df['%K'].shift(1) > df['%D'].shift(1)) &
    (df['%K'] < df['%D']) |
    (df['AD_slope'] < 0) |
    (df['VIX'] > 25)
)

capital = 100000
position = 0
entry_price = 0
equity_curve = []

for i in range(len(df)):
    price = df['Close'].iloc[i]

    if position == 0 and df['Buy'].iloc[i]:
        position = capital / price
        entry_price = price
        capital = 0

    elif position > 0:
        stop_loss = entry_price * 0.92

        if df['Sell'].iloc[i] or price < stop_loss:
            capital = position * price
            position = 0

    equity = capital if position == 0 else position * price
    equity_curve.append(equity)

df['Equity'] = equity_curve

returns = df['Equity'].pct_change().dropna()

total_return = (df['Equity'].iloc[-1] / df['Equity'].iloc[0] - 1) * 100
sharpe = np.sqrt(252) * returns.mean() / returns.std()
max_dd = ((df['Equity'] / df['Equity'].cummax()) - 1).min() * 100

print(f"Total Return: {total_return:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}%")

//stochastic oscillator: momentum indicator based on the idea that in an uptrend, price tends to close near the high of the range and in a downtrend, price tends to close near the low of the range.
//Accumulation / Distribution: volume based indicator that shows whether an asset is being accumulated (buying pressure) or distributed (selling pressure) over time. only its slope matters and not absolute value.
// VIX: volatility index that measures the market’s expected volatility for the next 30 days, derived from index option prices
// Role played by each: stochastic oscillator helps enter/exit att the right time. A/D line helps increase reliability as momentum without volume isn't accurste. VIX controls when strategy is allowed to trade depending on market volatility.
// stop loss was set at 8% because BTC's natural volatilty is 3-6%. most crytpto strategies use 6-10%.