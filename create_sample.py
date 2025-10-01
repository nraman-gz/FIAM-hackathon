import pandas as pd

print("Step 1: Counting observations per stock...")
# First pass: count observations per stock using chunks
obs_per_stock = {}
unique_dates = set()

for chunk in pd.read_csv("/Users/nikhil/Documents/FIAM-data/ret_sample.csv",
                         usecols=['id', 'date'],
                         chunksize=500000,
                         low_memory=False):
    unique_dates.update(chunk['date'].unique())
    counts = chunk.groupby('id').size()
    for stock_id, count in counts.items():
        obs_per_stock[stock_id] = obs_per_stock.get(stock_id, 0) + count

max_time_periods = len(unique_dates)
min_required_obs = max_time_periods * 0.8

print(f"Total stocks: {len(obs_per_stock)}")
print(f"Max time periods: {max_time_periods}")
print(f"Min required observations (80%): {min_required_obs}")

# Filter stocks with sufficient data
stocks_with_sufficient_data = {stock_id for stock_id, count in obs_per_stock.items()
                               if count >= min_required_obs}

print(f"Stocks with sufficient data: {len(stocks_with_sufficient_data)}")

# Sample 1000 stocks
import random
random.seed(42)
sampled_stocks = set(random.sample(sorted(stocks_with_sufficient_data),
                                   min(1000, len(stocks_with_sufficient_data))))

print(f"Sampled stocks: {len(sampled_stocks)}")
print("Step 2: Filtering data for sampled stocks...")

# Second pass: filter and write data
first_chunk = True
for chunk in pd.read_csv("/Users/nikhil/Documents/FIAM-data/ret_sample.csv",
                         chunksize=500000,
                         low_memory=False):
    filtered_chunk = chunk[chunk['id'].isin(sampled_stocks)]
    if len(filtered_chunk) > 0:
        filtered_chunk.to_csv('sample.csv',
                            mode='w' if first_chunk else 'a',
                            header=first_chunk,
                            index=False)
        first_chunk = False
        print(f"Processed chunk, kept {len(filtered_chunk)} rows")

print("Done!")