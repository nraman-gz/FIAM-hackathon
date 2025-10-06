import pandas as pd
import numpy as np
import os
import torch
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
import matplotlib.pyplot as plt

work_dir = "/teamspace/studios/this_studio/"

# Load data
file_path = os.path.join(work_dir, "us_stocks_data.parquet")
data = pd.read_parquet(file_path)

# Read predictors
file_path = os.path.join(work_dir, "factor_char_list.csv")
stock_vars = list(pd.read_csv(file_path)["variable"].values)

data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")

# Fill missing entries with cross-sectional median
data["year_month"] = data["date"].dt.to_period("M")
print("Imputing missing values...")
for var in stock_vars:
    if var in data.columns:
        data[var] = data.groupby("year_month")[var].transform(lambda x: x.fillna(x.median()))

data = data.dropna(subset=stock_vars)

# Generate time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# ITERATION 0 window
starting = pd.to_datetime("20050101", format="%Y%m%d")
counter = 0  # Iteration 0

cutoff = [
    starting + pd.DateOffset(years=counter),
    starting + pd.DateOffset(years=8 + counter),
    starting + pd.DateOffset(years=10 + counter),
    starting + pd.DateOffset(years=11 + counter),
]

train_mask = (data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])
val_mask = (data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])
test_mask = (data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])

train_data = data.loc[train_mask].copy()
val_data = data.loc[val_mask].copy()
test_data = data.loc[test_mask].copy()

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

train_max = train_data["time_idx"].max()

# Create dataset
training = TimeSeriesDataSet(
    data=train_data,
    time_idx="time_idx",
    target="stock_ret",
    group_ids=["gvkey"],
    min_encoder_length=3,
    max_encoder_length=24,
    min_prediction_length=1,
    max_prediction_length=12,
    time_varying_unknown_reals=["stock_ret"],
    time_varying_known_reals=stock_vars,
    allow_missing_timesteps=True,
)

# Load checkpoint
checkpoint_path = "/teamspace/studios/this_studio/checkpoints_deepar/iter_0/deepar-epoch=28-val_loss=67693776.00.ckpt"
print(f"Loading model from: {checkpoint_path}")
best_deepar = DeepAR.load_from_checkpoint(checkpoint_path)

# Create test dataset
train_val_data = pd.concat([train_data, val_data], ignore_index=True)
val_max = train_val_data["time_idx"].max()

test_dataset = TimeSeriesDataSet.from_dataset(
    training,
    train_val_data,
    predict=True,
    stop_randomization=True
)

test_dataloader = test_dataset.to_dataloader(
    train=False, batch_size=2048, num_workers=8
)

print("Generating predictions...")
torch.cuda.empty_cache()

test_predictions = best_deepar.predict(test_dataloader, mode="prediction", return_x=True, trainer_kwargs=dict(accelerator="gpu"))

# Extract predictions
prediction_index = test_dataset.x_to_index(test_predictions.x)

if test_predictions.output.ndim == 3:
    preds = test_predictions.output[:, :, 0].cpu().numpy()
elif test_predictions.output.ndim == 2:
    preds = test_predictions.output.cpu().numpy()
else:
    raise ValueError(f"Unexpected shape: {test_predictions.output.shape}")

print(f"Predictions shape: {preds.shape}")

# Create long format
rows = []
for i, gvkey in enumerate(prediction_index['gvkey'].values):
    for month_offset in range(12):
        rows.append({
            'gvkey': gvkey,
            'time_idx': val_max + 1 + month_offset,
            'pred_ret': preds[i, month_offset]
        })

test_pred_df = pd.DataFrame(rows)

# Merge with actuals
test_pred_df = test_pred_df.merge(
    test_data[["gvkey", "time_idx", "year", "month", "ret_eom", "stock_ret"]],
    on=["gvkey", "time_idx"],
    how="inner"
)

test_pred_df = test_pred_df[["gvkey", "year", "month", "ret_eom", "time_idx", "stock_ret", "pred_ret"]]

# Save
output_file = os.path.join(work_dir, "deepar_iter0_predictions.csv")
test_pred_df.to_csv(output_file, index=False)
print(f"\nSaved {len(test_pred_df)} predictions to {output_file}")

# Quick stats
mae = np.abs(test_pred_df['pred_ret'] - test_pred_df['stock_ret']).mean()
corr = test_pred_df[['pred_ret', 'stock_ret']].corr().iloc[0, 1]
print(f"MAE: {mae:.6f}")
print(f"Correlation: {corr:.6f}")

figures_dir = os.path.join(work_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

        # Plot predictions for a few random stocks
raw_predictions = best_deepar.predict(test_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="gpu"))

        # Use pytorch-forecasting's plot_prediction method
for idx in range(min(3, len(raw_predictions.x["decoder_lengths"]))):  # Plot first 3 stocks
    fig = best_deepar.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, show_future_observed=False)
    fig_path = os.path.join(figures_dir, f"iter{counter}_stock{idx}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
