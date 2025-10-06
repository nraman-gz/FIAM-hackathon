import pandas as pd
import numpy as np
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss
import logging
import optuna
import pickle

import matplotlib 
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepar_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.INFO)

# Global configuration
NUM_WORKERS = 24  # Increased for faster data loading
BATCH_SIZE = 16384  # Already maxed out

if __name__ == "__main__":

    work_dir = "/teamspace/studios/this_studio/"

    # read US stocks data (raw features, not PCA)
    file_path = os.path.join(work_dir, "us_stocks_data.parquet")
    data = pd.read_parquet(file_path)

    logger.info(f"Loaded data with shape: {data.shape}")

    # read list of predictors for stocks
    file_path = os.path.join(work_dir, "factor_char_list.csv")
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    logger.info(f"Using {len(stock_vars)} stock variables")

    data["date"] = pd.to_datetime(data["date"],format="%Y%m%d")

    # Fill missing entries with cross-sectional median for each month
    data["year_month"] = data["date"].dt.to_period("M")
    logger.info("Imputing missing values with cross-sectional medians...")
    for var in stock_vars:
        if var in data.columns:
            data[var] = data.groupby("year_month")[var].transform(lambda x: x.fillna(x.median()))

    # Drop variables that have missing values across all stocks for a given month
    logger.info(f"Rows before dropping NaNs: {len(data)}")
    data = data.dropna(subset=stock_vars)
    logger.info(f"Rows after dropping NaNs: {len(data)}")

    # Generate time index
    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()

    # Create dataframe to store results
    pred_out = pd.DataFrame()

    # sliding window estimation
    starting = pd.to_datetime("20050101", format="%Y%m%d")
    counter = 0

    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime(
        "20260101", format="%Y%m%d"
    ):
        logger.info(f"Starting iteration {counter}, date range: {starting} to {starting + pd.DateOffset(years=11)}")

        cutoff = [
            starting,
            starting
            + pd.DateOffset(
                years=8
            ),  # use 8 years as the training set (sliding window)
            starting
            + pd.DateOffset(
                years=10
            ),  # use the next 2 years as the validation set
            starting + pd.DateOffset(years=11),
        ]  # use the next year as the out-of-sample testing set

        logger.info(f"Cutoff dates - Train: {cutoff[0]} to {cutoff[1]}, Val: {cutoff[1]} to {cutoff[2]}, Test: {cutoff[2]} to {cutoff[3]}")

        # Convert cutoff dates to time_idx
        train_mask = (data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])
        val_mask = (data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])
        test_mask = (data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])

        train_data = data.loc[train_mask].copy()
        val_data = data.loc[val_mask].copy()
        test_data = data.loc[test_mask].copy()

        logger.info(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        train_max = train_data["time_idx"].max()

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

        validation = TimeSeriesDataSet.from_dataset(training, val_data, min_prediction_idx=train_max + 1)

        train_dataloader = training.to_dataloader(
            train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, batch_sampler="synchronized", pin_memory=True)

        val_dataloader = validation.to_dataloader(
            train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, batch_sampler="synchronized", pin_memory=True)

        logger.info(f"Created dataloaders with batch_size={BATCH_SIZE}")

        pl.seed_everything(42)

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=7, verbose=False, mode="min")

        # Create separate checkpoint directory for each iteration
        checkpoint_dir = os.path.join(work_dir, f"checkpoints_deepar/iter_{counter}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='deepar-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )

        # Hyperparameter optimization with Optuna (only for first iteration)
        study_path = os.path.join(work_dir, "checkpoints_deepar/iter_0/trial_1/deepar-epoch=00-val_loss=5199198720.00.ckpt")

        if counter == 0 and not os.path.exists(study_path):
            logger.info("Starting hyperparameter optimization with Optuna...")

            def objective(trial):
                # Suggest hyperparameters
                lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
                hidden_size = trial.suggest_int("hidden_size", 32, 128, step=32)
                rnn_layers = trial.suggest_int("rnn_layers", 1, 3)
                dropout = trial.suggest_float("dropout", 0.1, 0.4)

                logger.info(f"Trial {trial.number}: lr={lr:.6f}, hidden_size={hidden_size}, rnn_layers={rnn_layers}, dropout={dropout:.2f}")

                # Create model with suggested hyperparameters
                model = DeepAR.from_dataset(
                    training,
                    learning_rate=lr,
                    hidden_size=hidden_size,
                    rnn_layers=rnn_layers,
                    dropout=dropout,
                    loss=NormalDistributionLoss(),
                )

                # Create trial-specific checkpoint directory
                trial_checkpoint_dir = os.path.join(checkpoint_dir, f"trial_{trial.number}")
                os.makedirs(trial_checkpoint_dir, exist_ok=True)

                trial_checkpoint = ModelCheckpoint(
                    dirpath=trial_checkpoint_dir,
                    filename='deepar-{epoch:02d}-{val_loss:.2f}',
                    save_top_k=1,
                    monitor='val_loss',
                    mode='min'
                )

                trial_early_stop = EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=3,  # Lower patience for trials
                    verbose=False,
                    mode="min"
                )

                # Trainer for this trial
                trial_trainer = pl.Trainer(
                    max_epochs=3,  # Very fast trials with large batches on L40S
                    precision="32",  # FP32 to avoid NaN issues with PCA data
                    accelerator="gpu",
                    enable_model_summary=False,
                    gradient_clip_val=0.5,  # Increased gradient clipping
                    callbacks=[trial_early_stop, trial_checkpoint],
                    enable_checkpointing=True,
                    logger=False,  # Disable logging for trials
                )

                # Train
                trial_trainer.fit(
                    model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader
                )

                # Return validation loss
                val_loss = trial_trainer.callback_metrics.get('val_loss', float('inf'))

                return float(val_loss)

            # Create and run study - 10 trials × ~2-3 min = ~25 min total
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=10, show_progress_bar=True, n_jobs=1)

            logger.info("Hyperparameter optimization completed!")
            logger.info(f"Best trial value: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")

            # Save study
            with open(study_path, "wb") as f:
                pickle.dump(study, f)
            logger.info(f"Saved study to {study_path}")

            best_hparams = study.best_params

        elif os.path.exists(study_path):
            logger.info(f"Loading existing study from {study_path}")
            best_hparams = dict(DeepAR.load_from_checkpoint(study_path).hparams)
            logger.info(f"Loaded best hyperparameters: {best_hparams}")

        else:
            # Use default hyperparameters for subsequent iterations if no study exists
            best_hparams = {
                'learning_rate': 1e-3,
                'hidden_size': 64,
                'rnn_layers': 2,
                'dropout': 0.2
            }
            logger.info(f"Using default hyperparameters: {best_hparams}")

        # Create trainer for final model
        trainer = pl.Trainer(
            max_epochs=20,  # Reduced for speed - early stopping will kick in anyway
            precision="32",  # FP32 to avoid NaN issues with PCA data
            accelerator="gpu",
            enable_model_summary=True,
            gradient_clip_val=0.5,  # Increased gradient clipping
            callbacks=[early_stop_callback, checkpoint_callback],
            enable_checkpointing=True,
        )

        # Create final model with best hyperparameters
        deepar_model = DeepAR.from_dataset(
            training,
            learning_rate=best_hparams['learning_rate'],
            hidden_size=best_hparams['hidden_size'],
            rnn_layers=best_hparams['rnn_layers'],
            dropout=best_hparams['dropout'],
            loss=NormalDistributionLoss(),
        )

        # Check if checkpoint exists to resume training
        existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')] if os.path.exists(checkpoint_dir) else []
        ckpt_path = os.path.join(checkpoint_dir, existing_checkpoints[0]) if existing_checkpoints else None

        if ckpt_path:
            logger.info(f"Resuming iteration {counter} from checkpoint: {ckpt_path}")
        else:
            logger.info(f"Starting fresh training for iteration {counter}")

        logger.info(f"DeepAR config: {best_hparams}")
        logger.info(f"Starting training for iteration {counter}...")

        trainer.fit(
            deepar_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path
        )
        logger.info(f"Training completed for iteration {counter}")

        # load the best model according to the validation loss
        best_model_path = checkpoint_callback.best_model_path
        best_deepar = DeepAR.load_from_checkpoint(best_model_path)
        logger.info(f"Loaded best model from: {best_model_path}")

        # Create test dataset for predictions
        # Combine train and validation data to have full history before test
        train_val_data = pd.concat([train_data, val_data], ignore_index=True)
        val_max = train_val_data["time_idx"].max()

        # Create prediction dataset using predict=True mode
        test_dataset = TimeSeriesDataSet.from_dataset(
            training,
            train_val_data,  # Use train+val data as encoder
            predict=True,  # This creates prediction mode
            stop_randomization=True
        )
        # Use smaller batch size for prediction to avoid OOM
        test_dataloader = test_dataset.to_dataloader(
            train=False, batch_size=2048, num_workers=NUM_WORKERS  # Reduced from BATCH_SIZE
        )

        logger.info("Generating predictions...")

        # Clear GPU cache before prediction to avoid OOM
        import torch
        torch.cuda.empty_cache()

        # Make predictions - DeepAR outputs mean of normal distribution
        test_predictions = best_deepar.predict(test_dataloader, mode="prediction", return_x=True, trainer_kwargs=dict(accelerator="gpu"))

        # Extract predictions
        prediction_index = test_dataset.x_to_index(test_predictions.x)

        # DeepAR outputs: [num_stocks, 12 months, 2] where last dim is [mean, std]
        # We take the mean (index 0)
        if test_predictions.output.ndim == 3:
            # Shape: [num_stocks, 12 months, 2] - take mean
            preds = test_predictions.output[:, :, 0].cpu().numpy()
        elif test_predictions.output.ndim == 2:
            # Shape: [num_stocks, 12 months]
            preds = test_predictions.output.cpu().numpy()
        else:
            raise ValueError(f"Unexpected prediction shape: {test_predictions.output.shape}")

        logger.info(f"Generated predictions with shape: {preds.shape}")

        # Create long format dataframe
        rows = []
        for i, gvkey in enumerate(prediction_index['gvkey'].values):
            for month_offset in range(12):  # 12 month predictions
                rows.append({
                    'gvkey': gvkey,
                    'time_idx': val_max + 1 + month_offset,
                    'pred_ret': preds[i, month_offset]
                })

        test_pred_df = pd.DataFrame(rows)

        # Merge with actual stock_ret and date information from test_data
        test_pred_df = test_pred_df.merge(
            test_data[["gvkey", "time_idx", "year", "month", "ret_eom", "stock_ret"]],
            on=["gvkey", "time_idx"],
            how="inner"
        )

        # Rearrange columns so pred_ret is next to stock_ret
        test_pred_df = test_pred_df[["gvkey", "year", "month", "ret_eom", "time_idx", "stock_ret", "pred_ret"]]

        # Save prediction progress
        local_prediction_filename = os.path.join(work_dir, f"deepar_iter{counter}_predictions.csv")
        test_pred_df.to_csv(local_prediction_filename)
        logger.info(f"Saved predictions for iteration {counter} to {local_prediction_filename}")
        logger.info(f"Prediction summary: {len(test_pred_df)} predictions generated")

        # Create plots
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

        logger.info(f"Saved prediction plots to {figures_dir}/iter{counter}_stock*.png")

        # Append to overall output
        pred_out = pd.concat([pred_out, test_pred_df], ignore_index=True)

        counter += 1

    #Save all predictions to file
    out_path = os.path.join(work_dir, "deepar_output.csv")
    pred_out.to_csv(out_path, index=False)
    logger.info(f"Final output saved to {out_path}")
    logger.info(f"Total predictions: {len(pred_out)}")
