import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

output_dir = os.path.join('plots', 't1')
os.makedirs(output_dir, exist_ok=True)
print(f"Directory '{output_dir}' created (or already exists).")

print("Loading and preparing data...")
df = pd.read_csv('Stock.csv')

stock_symbol = 'AAPL'
df_stock = df[df['symbol'] == stock_symbol].copy()

df_stock['date'] = pd.to_datetime(df_stock['date'])
df_stock = df_stock.sort_values('date')
df_stock = df_stock.set_index('date')

ts_data = df_stock['close'].asfreq('B').fillna(method='ffill')

print("Generating Decomposition Plot...")
decomposition = seasonal_decompose(ts_data, model='additive', period=20)
fig = decomposition.plot()
fig.set_size_inches(12, 10)
plot_path = os.path.join(output_dir, 'decomposition.png')
plt.savefig(plot_path)
plt.close()
print(f"Saved: {plot_path}")

print("Generating Smoothing Plot...")
df_smooth = ts_data.to_frame()
df_smooth['Moving_Avg'] = ts_data.rolling(window=20).mean()
fit_es = ExponentialSmoothing(ts_data, trend='add', seasonal=None).fit()
df_smooth['Exp_Smoothing'] = fit_es.fittedvalues

plt.figure(figsize=(12, 6))

# Plot last 300 days
subset = ts_data[-300:]
subset_smooth = df_smooth[-300:]
plt.plot(subset.index, subset, label='Original', alpha=0.5)
plt.plot(subset_smooth.index, subset_smooth['Moving_Avg'], label='20-Day MA', color='orange')
plt.plot(subset_smooth.index, subset_smooth['Exp_Smoothing'], label='Exp Smoothing', color='green', linestyle='--')
plt.title(f'{stock_symbol} Smoothing Techniques (Last 300 Days)')
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_dir, 'smoothing.png')
plt.savefig(plot_path)
plt.close()
print(f"Saved: {plot_path}")

print("Training Model and Generating Forecast Plot...")
train_size = int(len(ts_data) * 0.9)
train, test = ts_data[:train_size], ts_data[train_size:]

model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
model_fit = model.fit(disp=False)
forecast = model_fit.forecast(steps=len(test))

rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(train.index[-100:], train[-100:], label='Train Data (Last 100 days)')
plt.plot(test.index, test, label='Actual Test Data', color='green')
plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='--')
plt.title(f'SARIMA Forecast (RMSE: {rmse:.2f})')
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_dir, 'forecast.png')
plt.savefig(plot_path)
plt.close()
print(f"Saved: {plot_path}")

print("\nAll tasks completed successfully.")