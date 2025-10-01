import pandas as pd

df = pd.read_parquet("ret_sample.parquet")

sample_stocks = df['id'].drop_duplicates().sample(n=1000, random_state=42)
sample_df = df[df['id'].isin(sample_stocks)]

sample_df.to_csv('sample.csv')