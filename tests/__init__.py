import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Fetch 9 days of 1-minute OHLC data for BTC
end = datetime.utcnow()
start = end - timedelta(days=9)
df = yf.download("BTC-USD", start=start, end=end, interval="1m", progress=False)

# Format: timestamp,open,high,low,close
df_output = df.reset_index()
df_output['Datetime'] = df_output['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_output = df_output[['Datetime', 'Open', 'High', 'Low', 'Close']]
df_output.columns = ['t', 'o', 'h', 'l', 'c']

# Save to historical_data.csv in project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(project_root, 'historical_data.csv')
df_output.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
print(f"Rows: {len(df_output)}")
print(df_output.head(10))
