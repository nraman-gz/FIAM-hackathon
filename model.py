import pandas as pd
import numpy as np
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TimeSeriesDataSet, Baseline, DeepAR
from pytorch_forecasting.metrics import MAE, RMSE, MultivariateNormalDistributionLoss

if __name__ == "__main__":
    work_dir = "/Users/nikhil/Documents/FIAM-hackathon"

    # read sample data
    file_path = os.path.join(work_dir, "sample.csv")
    data = pd.read_csv(file_path, low_memory=False)

    # read list of predictors for stocks
    file_path = os.path.join(work_dir, "factor_char_list.csv")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    # Keep only 20% of variables for testing
    np.random.seed(42)
    n_keep = int(len(stock_vars) * 0.5)
    stock_vars = np.random.choice(stock_vars, size=n_keep, replace=False).tolist()

    data["date"] = pd.to_datetime(data["date"],format="%Y%m%d")

    # Fill missing entries with cross-sectional median for each month
    data["year_month"] = data["date"].dt.to_period("M")
    for var in stock_vars:
        if var in data.columns:
            data[var] = data.groupby("year_month")[var].transform(lambda x: x.fillna(x.median()))

    # Drop variables that have missing values across all stocks for a given month
    data = data.dropna(subset=stock_vars)

    # Generate time index
    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()

    train_max = data["time_idx"].max() - 24 # 2 Years

    training = TimeSeriesDataSet(
        data=data[lambda x: x.time_idx <= train_max],
        time_idx="time_idx",
        target="stock_ret",
        group_ids=["id"],
        min_encoder_length=12,
        max_encoder_length=24,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_unknown_reals=["stock_ret"],
        time_varying_known_reals=stock_vars,
        allow_missing_timesteps=True,
    )


    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=train_max + 1)
    batch_size = 128

    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=7, batch_sampler="synchronized", 
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=7, batch_sampler="synchronized"
    )

    # calculate baseline absolute error
    baseline_predictions = Baseline().predict(val_dataloader, trainer_kwargs=dict(accelerator="mps"), return_y=True)
    baselineRMSE = RMSE()(baseline_predictions.output, baseline_predictions.y)

    print(baselineRMSE)

    pl.seed_everything(42)

    trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=1e-1)
    net = DeepAR.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=30,
    rnn_layers=2,
    loss=MultivariateNormalDistributionLoss(rank=30),
    optimizer="Adam",
    )

    
    

