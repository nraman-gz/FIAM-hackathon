import pandas as pd

# Read the two parquet files
train_df = pd.read_parquet("data processing PCA/cleaned data/cleaned_train_financial_data_with_pca.parquet")
test_df = pd.read_parquet("data processing PCA/cleaned data/cleaned_test_financial_data_with_pca.parquet")

combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Save to a new parquet file
combined_df.to_parquet("data processing PCA/cleaned data/combined_financial_data_with_pca.parquet", index=False)
