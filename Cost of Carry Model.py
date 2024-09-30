
# Install the following on your terminal before using price loaders:

# pip install git+https://github.com/batprem/price-loaders.git - github for recall data
# pip install --upgrade pip
# pip install filterpy
# pip install plotly
# pip install pmdarima
# pip install --upgrade pandas


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go

from price_loaders.tradingview import load_asset_price
from price_loaders.tradingview import load_raw_data

spots_df = load_asset_price(
  "CL1!", # Asset symbol referenced in TradingView
  5000, # 5000 candle look back
  "5", # Timeframe (5 minute per candle)
  None # Timezone default to Thai timezone (+07:00)
)

futures_df = load_asset_price(
  "CLZ2024", # Asset symbol referenced in TradingView
  5000, # 5000 candle look back
  "5", # Timeframe (5 minutes per candle)
  None # Timezone default to Thai timezone (+07:00)
)



# Adding expiration date for future contracts
futures_df['expiry_date'] = futures_df['time'] + pd.DateOffset(months=2)

# Creating new df based on futures_df and spots_df
combined_df = pd.merge(spots_df, futures_df, on='time', suffixes=('_spot', '_futures'))

# Adding time to maturity 
combined_df['time_to_maturity'] = (combined_df['expiry_date'] - combined_df['time']).dt.days / 365.0

# Adding risk free rate
combined_df['risk_free_rate'] = 0.02

final_df = combined_df[['time', 'close_spot', 'close_futures', 'time_to_maturity', 'risk_free_rate']]
final_df = final_df.rename(columns={'close_spot': 'spot_price', 'close_futures': 'futures_price'})
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)




# Define the 2500th cell index
start_index = 2500

# Slice the DataFrame up to the 2500th cell
df_train = final_df.iloc[:start_index]

# Extract required columns
spot_prices = df_train['spot_price'].values
futures_prices = df_train['futures_price'].values
time_to_maturity = df_train['time_to_maturity'].values
risk_free_rate = df_train['risk_free_rate'].values[0]  # Assuming constant rate for simplicity

# Calculate Convenience Yield 
convenience_yield = risk_free_rate - np.log(futures_prices / spot_prices) / time_to_maturity

# Forecast future prices from the 2500th cell
forecast_length = len(final_df) - start_index
forecasted_futures_prices = []

# Initialize forecast with the last known futures price
current_futures_price = final_df.iloc[start_index]['futures_price']

# Loop for forecasting
for i in range(forecast_length):
    last_spot_price = final_df.iloc[start_index + i]['spot_price']
    last_time_to_maturity = final_df.iloc[start_index + i]['time_to_maturity']
    
    # Forecast next futures price
    next_futures_price = last_spot_price * np.exp((risk_free_rate - convenience_yield[start_index - 1]) * last_time_to_maturity)
    
    forecasted_futures_prices.append(next_futures_price)
    
    # Update convenience yield
    if i < len(convenience_yield) - 1:
        convenience_yield[start_index - 1] = risk_free_rate - np.log(next_futures_price / last_spot_price) / last_time_to_maturity

# Create a DataFrame for the forecasted futures prices
forecast_dates = final_df.iloc[start_index:].reset_index(drop=True)['time']
forecast_df = pd.DataFrame({
    'time': forecast_dates,
    'forecasted_futures_price': forecasted_futures_prices
})

# Display forecast DataFrame
print(forecast_df)


# Merge forecasted prices with actual futures prices to a new df
merged_df = pd.merge(final_df[['time', 'futures_price']], forecast_df, on='time', how='outer')

# Plot 
plt.figure(figsize=(12, 6))
plt.plot(merged_df['time'], merged_df['futures_price'], label='Actual Futures Price', color='blue')
plt.plot(merged_df['time'], merged_df['forecasted_futures_price'], label='Forecasted Futures Price', color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Actual vs Forecasted Futures Price')
plt.legend()
plt.grid(True)
plt.show()

# Create an interactive plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=merged_df['time'],
    y=merged_df['futures_price'],
    mode='lines',
    name='Actual Futures Price',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=merged_df['time'],
    y=merged_df['forecasted_futures_price'],
    mode='lines',
    name='Forecasted Futures Price',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='Actual vs Forecasted Futures Price',
    xaxis_title='Time',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True
)

fig.show()

