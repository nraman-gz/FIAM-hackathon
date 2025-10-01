import pandas as pd
import numpy as np
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TimeSeriesDataSet, LSTM, TemporalFusionTransformer
from pytorch_forecasting.metrics import MASE


work_dir = "/Users/nikhil/Documents/FIAM-hackathon"


# read sample data
file_path = os.path.join(work_dir, "sample.csv") 
data = pd.read_csv(file_path, low_memory=False)

# read list of predictors for stocks
file_path = os.path.join(work_dir, "factor_char_list.csv")
stock_vars = list(pd.read_csv(file_path)["variable"].values)

original_count = len(stock_vars)
stock_vars = [var for var in stock_vars if var in data.columns and data[var].notna().all()]
dropped_count = original_count - len(stock_vars)
print(f"Dropped {dropped_count} variables with missing values. {len(stock_vars)} variables remaining.")

data["date"] = pd.to_datetime(data["date"],format="%Y%m%d")

train_max = pd.to_datetime("20230630", format="%Y%m%d")

train_data = data[data['date'] <= train_max]
train_data["time_idx"] = train_data["date"].dt.year * 12 + train_data["date"].dt.month
train_data["time_idx"] -= train_data["time_idx"].min()


TSdata = TimeSeriesDataSet(
    data=train_data,
    time_idx="time_idx",
    target="stock_ret",
    group_ids=["id"],
    min_encoder_length=12,
    max_encoder_length=24,
    min_prediction_length=1,
    max_prediction_length=1,
    time_varying_unknown_reals=["stock_ret"] + stock_vars,
    allow_missing_timesteps=True
)

# Create dataloaders
train_dataloader = TSdata.to_dataloader(train=True, batch_size=64, num_workers=0)

# Initialize LSTM model
lstm_model = LSTM.from_dataset(
    TSdata,
    hidden_size=50,
    rnn_layers=2,
    dropout=0.1,
    learning_rate=0.001,
    reduce_on_plateau_patience=4
)

# Setup trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="auto",
    callbacks=[
        EarlyStopping(monitor="train_loss", patience=3),
        LearningRateMonitor()
    ]
)

# Train the model
trainer.fit(lstm_model, train_dataloaders=train_dataloader)

# Prepare test data
test_data = data[data['date'] > train_max].copy()
test_data["time_idx"] = test_data["date"].dt.year * 12 + test_data["date"].dt.month
test_data["time_idx"] -= train_data["time_idx"].min()

# Create test dataset
test_dataset = TimeSeriesDataSet.from_dataset(TSdata, test_data, predict=True, stop_randomization=True)
test_dataloader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

# Generate predictions
predictions = lstm_model.predict(test_dataloader, mode="prediction", return_x=True)

# Calculate MSE
actuals = predictions.x["decoder_target"].numpy()
preds = predictions.output.numpy()
mse = np.mean((actuals - preds) ** 2)

print(f"Out-of-sample MSE: {mse:.6f}")

