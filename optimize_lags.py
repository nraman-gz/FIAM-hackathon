import pandas as pd
import numpy as np
import os
import xgboost as xgb
from itertools import combinations

work_dir = "/teamspace/studios/this_studio/"

# Load US stocks data (RAW FEATURES)
file_path = os.path.join(work_dir, "us_stocks_data.parquet")
data = pd.read_parquet(file_path)

# Read list of raw features
file_path = os.path.join(work_dir, "factor_char_list.csv")
stock_vars = list(pd.read_csv(file_path)["variable"].values)

data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")

# Fill missing entries with cross-sectional median
data["year_month"] = data["date"].dt.to_period("M")
for var in stock_vars:
    if var in data.columns:
        data[var] = data.groupby("year_month")[var].transform(lambda x: x.fillna(x.median()))

# Drop rows with remaining NaNs
data = data.dropna(subset=stock_vars)

# Clean stock returns
data['stock_ret'] = np.clip(data['stock_ret'], -1.0, 1.0)

# Generate time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# Sort by gvkey and time for lag calculation
data = data.sort_values(['gvkey', 'time_idx']).reset_index(drop=True)

# Test different lag configurations
lag_options = [1, 3, 6, 12]
results = []

# Test all combinations of lags (including no lags as baseline)
all_configs = [[]]  # Start with no lags
for r in range(1, len(lag_options) + 1):
    for combo in combinations(lag_options, r):
        all_configs.append(list(combo))

# Use iteration 0 for testing
starting = pd.to_datetime("20050101", format="%Y%m%d")
cutoff = [
    starting,
    starting + pd.DateOffset(years=8),  # Train: 2005-2013
    starting + pd.DateOffset(years=10), # Val: 2013-2015
    starting + pd.DateOffset(years=11), # Test: 2015-2016
]

train_mask = (data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])
val_mask = (data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])
test_mask = (data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])

for lags in all_configs:
    config_name = f"lags_{'-'.join(map(str, lags))}" if lags else "no_lags"
    print(f"\nTesting configuration: {config_name}")

    # Create a copy with lag features
    data_with_lags = data.copy()
    lag_features = []

    for lag in lags:
        lag_col = f'ret_lag{lag}'
        data_with_lags[lag_col] = data_with_lags.groupby('gvkey')['stock_ret'].shift(lag)
        lag_features.append(lag_col)

    # Combine base features with lag features
    all_features = stock_vars + lag_features

    # Split data
    train_data = data_with_lags.loc[train_mask].copy()
    val_data = data_with_lags.loc[val_mask].copy()
    test_data = data_with_lags.loc[test_mask].copy()

    # Drop rows with NaN in lag features
    if lag_features:
        train_data = train_data.dropna(subset=lag_features)
        val_data = val_data.dropna(subset=lag_features)
        test_data = test_data.dropna(subset=lag_features)

    # Prepare features
    X_train = train_data[all_features].values
    y_train = train_data['stock_ret'].values

    X_val = val_data[all_features].values
    y_val = val_data['stock_ret'].values

    X_test = test_data[all_features].values
    y_test = test_data['stock_ret'].values

    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20
    )

    # Train with validation
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Generate predictions
    test_preds = model.predict(X_test)

    # Calculate metrics
    test_pred_df = test_data[["gvkey", "year", "month", "ret_eom", "time_idx", "stock_ret"]].copy()
    test_pred_df["pred_ret"] = test_preds

    mae = np.abs(test_pred_df['pred_ret'] - test_pred_df['stock_ret']).mean()
    rmse = np.sqrt(((test_pred_df['pred_ret'] - test_pred_df['stock_ret']) ** 2).mean())
    corr = test_pred_df[['pred_ret', 'stock_ret']].corr().iloc[0, 1]

    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  CORRELATION: {corr:.6f}")

    results.append({
        'config': config_name,
        'lags': lags,
        'n_features': len(all_features),
        'mae': mae,
        'rmse': rmse,
        'correlation': corr
    })

# Summary
print("\n" + "="*60)
print("SUMMARY OF ALL CONFIGURATIONS")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('correlation', ascending=False)

for idx, row in results_df.iterrows():
    print(f"{row['config']:20s} | Corr: {row['correlation']:.6f} | MAE: {row['mae']:.6f} | Features: {row['n_features']}")

# Save results
output_file = os.path.join(work_dir, "lag_optimization_results.csv")
results_df.to_csv(output_file, index=False)
print(f"\nSaved results to {output_file}")

# Best configuration
best_config = results_df.iloc[0]
print("\n" + "="*60)
print("BEST CONFIGURATION")
print("="*60)
print(f"Config: {best_config['config']}")
print(f"Lags: {best_config['lags']}")
print(f"Correlation: {best_config['correlation']:.6f}")
print(f"MAE: {best_config['mae']:.6f}")
print(f"RMSE: {best_config['rmse']:.6f}")
print("="*60)
