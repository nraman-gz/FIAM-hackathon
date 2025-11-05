import pandas as pd
import warnings, os
warnings.simplefilter("ignore")

'''
this script merges sentiment data with stock price data based on nearest previous date
it also filters stocks that are not in USA or have intrinsic_value <= 500mm
'''
dataframe_sentiment = pd.read_csv("/home/dave/Desktop/code_hackathon/sentiment_result/processed_sentiment.csv")
stock_df = pd.read_csv("/home/dave/Desktop/code_hackathon/ret_sample.csv")
gvkeys = stock_df[["date","gvkey","prc"]][(stock_df["intrinsic_value"]>500) & (stock_df['excntry'] == 'USA')]['gvkey'].unique()
for gvkey in gvkeys:
    if not gvkey in dataframe_sentiment['gvkey'].values:
        continue
    df_a = stock_df[["date","gvkey","prc","ret_1_0"]][(stock_df["intrinsic_value"]>500) & (stock_df["gvkey"]==gvkey)]
    df_b = dataframe_sentiment[dataframe_sentiment["chunks"]>5][(dataframe_sentiment["gvkey"]==gvkey) & (dataframe_sentiment["delta_score"]!=0)]
        # Convert dates to datetime if not already
    df_a['date'] = pd.to_datetime(df_a['date'], format='%Y%m%d')
    df_b['date'] = pd.to_datetime(df_b['date'], format='%Y%m%d')

    # Sort both dataframes by date
    df_a = df_a.sort_values('date')
    df_b = df_b.sort_values('date')

    # Merge with merge_asof to get nearest previous date
    merged_df = pd.merge_asof(
        df_a,
        df_b,
        on='date',
        by='gvkey',
        direction='backward',
        suffixes=('_price', '_sentiment')
    )

    # Filter out rows where the merged date is null (no matching sentiment)
    merged_df = merged_df.dropna(subset=['date'])
    merged_df.to_csv(f"/home/dave/Desktop/code_hackathon/merged_data/merged_{gvkey}.csv", index=False)