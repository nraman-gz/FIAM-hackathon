import pandas as pd
import numpy as np
import os
import xgboost as xgb
import pickle
from pykalman import KalmanFilter

def kalman_smooth_stock(group):
            """Apply Kalman filter to smooth predictions for a single stock"""
            if len(group) < 2:
                group['pred_ret'] = group['pred_ret_raw']
                return group

            # Simple Kalman filter for smoothing
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=group['pred_ret_raw'].iloc[0],
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=0.01
            )

            # Smooth the predictions
            state_means, _ = kf.filter(group['pred_ret_raw'].values)
            group['pred_ret'] = state_means.flatten()

            return group


if __name__ == "__main__":

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

    # Clean stock returns - clip extreme outliers
    data['stock_ret'] = np.clip(data['stock_ret'], -1.0, 1.0)

    # Generate time index
    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()

    # Sort by gvkey and time for lag calculation
    data = data.sort_values(['gvkey', 'time_idx']).reset_index(drop=True)

    # Create lag features (1-12 months)
    lag_features = []
    for lag in range(1, 13):
        lag_col = f'ret_lag{lag}'
        data[lag_col] = data.groupby('gvkey')['stock_ret'].shift(lag)
        lag_features.append(lag_col)

    # Combine all features
    all_features = stock_vars + lag_features

    # Create dataframe to store results
    pred_out = pd.DataFrame()

    # Sliding window estimation
    starting = pd.to_datetime("20050101", format="%Y%m%d")
    counter = 0

    while (starting + pd.DateOffset(years=11)) <= pd.to_datetime(
        "20260701", format="%Y%m%d"
    ):
        cutoff = [
            starting,
            starting + pd.DateOffset(years=8),  # 8 years training
            starting + pd.DateOffset(years=10),  # 2 years validation
            starting + pd.DateOffset(years=11),  # 1 year test
        ]

        # Convert cutoff dates to masks
        train_mask = (data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])
        val_mask = (data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])
        test_mask = (data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])

        train_data = data.loc[train_mask].copy()
        val_data = data.loc[val_mask].copy()
        test_data = data.loc[test_mask].copy()

        # Drop rows with NaN in lag features (first 12 months per stock won't have all lags)
        train_data = train_data.dropna(subset=lag_features)
        val_data = val_data.dropna(subset=lag_features)
        test_data = test_data.dropna(subset=lag_features)

        # Prepare features and target
        X_train = train_data[all_features].values
        y_train = train_data['stock_ret'].values

        X_val = val_data[all_features].values
        y_val = val_data['stock_ret'].values

        X_test = test_data[all_features].values
        y_test = test_data['stock_ret'].values

        # Enable gpu
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            device = 'cuda',
            early_stopping_rounds=20
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Save model
        model_path = os.path.join(work_dir, f"checkpoints_xgb_lags/iter_{counter}")
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, "model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Generate predictions on test set
        test_preds = model.predict(X_test)

        # Create predictions dataframe
        test_pred_df = test_data[["gvkey", "year", "month", "ret_eom", "time_idx", "stock_ret"]].copy()
        test_pred_df["pred_ret_raw"] = test_preds

        # Apply Kalman filtering per stock
        test_pred_df = test_pred_df.sort_values(['gvkey', 'time_idx']).groupby('gvkey', group_keys=False).apply(kalman_smooth_stock)

        # Save prediction progress
        save_path = os.path.join(work_dir, "xgb_results/")
        local_prediction_filename = os.path.join(save_path, f"xgb{counter}_predictions.csv")
        test_pred_df.to_csv(local_prediction_filename, index=False)

        # Append to overall output
        pred_out = pd.concat([pred_out, test_pred_df], ignore_index=True)

        counter += 1

        # Move to next window
        starting = starting + pd.DateOffset(years=1)

    # Save all predictions to file
    out_path = os.path.join(save_path, "xgb_output.csv")
    pred_out.to_csv(out_path, index=False)

    # Overall statistics
    overall_mae = np.abs(pred_out['pred_ret'] - pred_out['stock_ret']).mean()
    overall_rmse = np.sqrt(((pred_out['pred_ret'] - pred_out['stock_ret']) ** 2).mean())
    overall_corr = pred_out[['pred_ret', 'stock_ret']].corr().iloc[0, 1]

    print(f"MAE: {overall_mae}"})