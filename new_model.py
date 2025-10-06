import pandas as pd
import numpy as np
import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, Baseline, DeepAR, LSTM, TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import optuna, statsmodels, pickle
from pytorch_forecasting.metrics import MAE, RMSE, MultivariateNormalDistributionLoss, QuantileLoss
import logging

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optuna_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set optuna logging to be verbose
optuna.logging.set_verbosity(optuna.logging.INFO)

def hidden_size_search(model,sizes, data, train_dataloaders, val_dataloaders):

    hidden_sizes = sizes
    best_val_loss = float('inf')
    best_config = None

    for hidden_size in hidden_sizes:
        search_model = model.from_dataset(
            data,
            learning_rate=0.03,
            hidden_size=hidden_size,
            hidden_continuous_size=hidden_size // 2,
        )

    trainer.fit(search_model, train_dataloaders, val_dataloaders)

    val_loss = trainer.callback_metrics['val_loss'].item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_config = hidden_size

    return best_config


if __name__ == "__main__":

    work_dir = "/teamspace/studios/this_studio/"

    # read sample data
    file_path = os.path.join(work_dir, "cleaned_PCA_data.parquet")
    study_path = os.path.join(work_dir, "checkpoints/iter_0/trial_0")
    data = pd.read_parquet(file_path)

    logger.info(f"Loaded data with shape: {data.shape}")

    # read list of predictors for stocks
    file_path = os.path.join(work_dir, "features.csv")
    stock_vars = list(pd.read_csv(file_path)["feature"].values)

    logger.info(f"Using {len(stock_vars)} stock variables")

    data["date"] = pd.to_datetime(data["date"],format="%Y%m%d")

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

        batch_size = 6144

        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=8, persistent_workers = True,batch_sampler="synchronized")
        
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=8, persistent_workers = True,batch_sampler="synchronized")

        logger.info(f"Created dataloaders with batch_size={batch_size}")

        pl.seed_everything(42)

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")

        # Create separate checkpoint directory for each iteration
        checkpoint_dir = os.path.join(work_dir, f"checkpoints/iter_{counter}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='tft-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )

        # Check if we're resuming from iteration 0 checkpoint
        existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')] if os.path.exists(checkpoint_dir) else []

        trainer = pl.Trainer(
            max_epochs=15,
            precision = "32",
            profiler = "simple",
            accelerator="gpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            # limit_train_batches=50,
            enable_checkpointing=True,
        )


        tft_model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate = 1e-3,
            hidden_size = 8,
            attention_head_size = 4,
            dropout = 0.1,
            hidden_continuous_size = 8,
            loss = QuantileLoss()
        )

        # Tune hyperparameters (batch size, learning rate, hidden layer size)
        if os.path.exists(study_path):
            logger.info(f"Loading existing study from {study_path}")
            chkpt_path = "checkpoints/iter_0/trial_0/epoch=1.ckpt"
            chkpt_model = TemporalFusionTransformer.load_from_checkpoint(chkpt_path)
            best_hparams = dict(chkpt_model.hparams)
            logger.info(f"Loaded best hyperparameters: {best_hparams}")

        # elif counter == 0:
        #     logger.info("Starting hyperparameter optimization...")
        #     logger.info("Search spaces:")
        #     logger.info("  - hidden_size: (64, 256)")
        #     logger.info("  - learning_rate: (1e-5, 1e-1)")
        #     logger.info("  - attention_head_size: (1, 4)")
        #     logger.info("  - dropout: (0.1, 0.3)")
        #     logger.info("  - n_trials: 5")
            
        #     # tuner = Tuner(trainer)
        #     # tuned_batch_size = tuner.scale_batch_size(tft_model, 
        #     # train_dataloaders=train_dataloader,
        #     # val_dataloaders=val_dataloader)
        #     # optimal_batch_size = tuned_batch_size if tuned_batch_size is not None else batch_size
        #     # hardcode for now, does not work
        #     # optimal_lr = tuner.lr_find(tft_model,
        #     #               train_dataloaders=train_dataloader,
        #     #               val_dataloaders=val_dataloader,
        #     #               min_lr = 1e-6,
        #     #               max_lr=1e-2)

        #     # optimal_hidden_size = hidden_size_search(sizes=[8,16,32,64],
        #     #                                          data = training,
        #     #                                          train_dataloaders=train_dataloader,
        #     #                                          val_dataloaders=val_dataloader)

        #     study = optimize_hyperparameters(
        #         train_dataloaders=train_dataloader,
        #         val_dataloaders=val_dataloader,
        #         model_path=checkpoint_dir,
        #         hidden_size_range=(64,256),
        #         learning_rate_range=(1e-5, 1e-1),
        #         attention_head_size_range=(1, 4),
        #         dropout_range=(0.1, 0.3),
        #         n_trials=5,
        #         trainer_kwargs=dict(max_epochs=15)  
        #     )

        #     logger.info("Hyperparameter optimization completed!")
        #     logger.info(f"Best trial value: {study.best_value}")
        #     logger.info(f"Best parameters: {study.best_params}")
            
        #     # Save detailed study results
        #     study_df = study.trials_dataframe()
        #     study_df.to_csv(os.path.join(work_dir, "detailed_study_results.csv"), index=False)
        #     logger.info("Saved detailed study results to detailed_study_results.csv")

        #     # Save study results to load for later iterations
        #     study_path_pkl = os.path.join(work_dir, "test_study.pkl")
        #     with open(study_path_pkl, "wb") as fout:
        #         pickle.dump(study, fout)
        #     logger.info(f"Saved study to {study_path_pkl}")

        #     best_hparams = study.best_params

        # else:
        #     logger.info(f"Loading study results for iteration {counter}")
        #     # Load study results from first iteration
        #     study_path_pkl = os.path.join(work_dir, "test_study.pkl")
        #     with open(study_path_pkl, "rb") as fin:
        #         study = pickle.load(fin)
        #         best_hparams = study.best_params
        #     logger.info(f"Using best hyperparameters from iteration 0: {best_hparams}")

        logger.info(f"Final hyperparameters for training: {best_hparams}")

        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size,num_workers=8,persistent_workers = True,batch_sampler="synchronized")
        
        val_dataloader = validation.to_dataloader(
            train=False,batch_size=batch_size,num_workers=8,persistent_workers = True,batch_sampler="synchronized")

        tft_model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate = best_hparams['learning_rate'],
            hidden_size = best_hparams['hidden_size'],
            attention_head_size = best_hparams['attention_head_size'],
            dropout = best_hparams['dropout'],
            hidden_continuous_size = best_hparams['hidden_size'] // 2, # convention
            loss = QuantileLoss()
        )

        # Check if checkpoint exists to resume training
        existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        ckpt_path = os.path.join(checkpoint_dir, existing_checkpoints[0]) if existing_checkpoints else None

        if ckpt_path:
            logger.info(f"Resuming iteration {counter} from checkpoint: {ckpt_path}")

        logger.info(f"Starting training for iteration {counter}...")
        trainer.fit(
            tft_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path
        )
        logger.info(f"Training completed for iteration {counter}")

        # calculate baseline absolute error
        # baseline_predictions = Baseline().predict(val_dataloader, trainer_kwargs=dict(accelerator="mps"), return_y=True)
        # baselineRMSE = RMSE()(baseline_predictions.output, baseline_predictions.y)

        # print(baselineRMSE)

        # deepAR_model = DeepAR.from_dataset(
        #     training,
        #     learning_rate=1e-3,
        #     log_interval=10,
        #     log_val_interval=1,
        #     hidden_size=30,
        #     rnn_layers=2,
        #     optimizer="Adam",
        #     loss=MultivariateNormalDistributionLoss(rank=30),)
        # 

        # load the best model according to the validation loss
        # (given that we use early stopping, this is not necessarily the last epoch)
        best_model_path = checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        logger.info(f"Loaded best model from: {best_model_path}")

        # Create test dataset for predictions
        # Need to use train+val data as encoder to predict test period
        # Test period comes AFTER validation period

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
        test_dataloader = test_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=8
        )

        logger.info("Generating predictions...")
        # Make predictions - this will predict max_prediction_length (12 months) ahead
        test_predictions = best_tft.predict(test_dataloader, mode="prediction", return_x=True, trainer_kwargs=dict(accelerator="gpu"))

        # Extract predictions - with predict=True, we get [num_stocks, 12_months, 7_quantiles]
        # We need to reshape this into a long format with one row per stock-month
        prediction_index = test_dataset.x_to_index(test_predictions.x)

        # Get predictions (median quantile)
        if test_predictions.output.ndim == 3:
            # Shape: [num_stocks, 12 months, 7 quantiles]
            preds = test_predictions.output[:, :, 3].cpu().numpy()  # Take median quantile
        elif test_predictions.output.ndim == 2:
            # Shape: [num_stocks, 12 months]
            preds = test_predictions.output.cpu().numpy()
        else:
            raise ValueError(f"Unexpected prediction shape: {test_predictions.output.shape}")

        logger.info(f"Generated predictions with shape: {preds.shape}")

        # Create long format dataframe
        # In predict=True mode, predictions start from val_max + 1 (test period)
        rows = []
        for i, stock_id in enumerate(prediction_index['id'].values):
            # Predictions start from val_max + 1 (first month of test period)
            for month_offset in range(12):  # 12 month predictions
                rows.append({
                    'id': stock_id,
                    'time_idx': val_max + 1 + month_offset,
                    'pred_ret': preds[i, month_offset]
                })

        test_pred_df = pd.DataFrame(rows)

        # Merge with actual stock_ret and date information from test_data
        test_pred_df = test_pred_df.merge(
            test_data[["id", "time_idx", "year", "month", "ret_eom", "stock_ret"]],
            on=["id", "time_idx"],
            how="inner"  # Changed to inner to only keep test period predictions
        )

        # Rearrange columns so pred_ret is next to stock_ret
        test_pred_df = test_pred_df[["id", "year", "month", "ret_eom", "time_idx", "stock_ret", "pred_ret"]]
        
        # Save prediction progress
        local_prediction_filename = os.path.join(work_dir, f"iter{counter}_predictions.csv")
        test_pred_df.to_csv(local_prediction_filename)
        logger.info(f"Saved predictions for iteration {counter} to {local_prediction_filename}")
        logger.info(f"Prediction summary: {len(test_pred_df)} predictions generated")

        # Append to overall output
        pred_out = pd.concat([pred_out, test_pred_df], ignore_index=True)

        counter += 1

    #Save all predictions to file
    out_path = os.path.join(work_dir, "output.csv")
    pred_out.to_csv(out_path, index=False)
    logger.info(f"Final output saved to {out_path}")
    logger.info(f"Total predictions: {len(pred_out)}")