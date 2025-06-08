import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- Correct Relative Import ---
# This assumes you run from the project root with `python -m src.evals.transformer_accuracy`
from ..transformer_based.pretrain import (
    PVTransformer, TimeseriesDataset, create_sequences, 
    load_and_process_data_per_station, CONFIG as PRETRAIN_CONFIG
)

# --- CONFIGURATION FOR EVALUATION ---
EVAL_CONFIG = {
    # Inherit data sources and model architecture from pre-training config
    **PRETRAIN_CONFIG,
    # Specify path to the trained model
    "model_path": "checkpoints/devout_wildflower/best.pth",
    "random_seed": 42,
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to {seed}")

def evaluate_model(config):
    set_seed(config["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # --- Load data per station using the refactored function ---
    processed_stations, scaler, feature_names = load_and_process_data_per_station(config)
    num_features = len(feature_names)

    model = PVTransformer(
        num_features=num_features, d_model=config["d_model"], nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"], num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"]
    ).to(device)

    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"Model file not found at: {config['model_path']}")
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()

    all_stations_actuals = []
    all_stations_predictions = []

    print("Starting evaluation on each station's test set...")
    for station_idx, (_, test_scaled, target_col_idx) in enumerate(processed_stations):
        X_test, y_test = create_sequences(test_scaled, config["input_seq_len"], config["output_seq_len"])
        test_dataset = TimeseriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
        
        station_predictions_scaled = []
        station_actuals_scaled = []

        with torch.no_grad():
            for src, tgt in test_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_for_pred = tgt[:, :-1, :]
                tgt_for_loss = tgt[:, 1:, target_col_idx]
                
                prediction = model(src, tgt_for_pred)
                
                station_predictions_scaled.append(prediction.cpu().numpy())
                station_actuals_scaled.append(tgt_for_loss.cpu().numpy())
        
        # Flatten and inverse transform this station's data
        actuals_flat_scaled = np.concatenate([s.flatten() for s in station_actuals_scaled])
        predictions_flat_scaled = np.concatenate([p.flatten() for p in station_predictions_scaled])
        
        dummy_actuals = np.zeros((len(actuals_flat_scaled), num_features))
        dummy_actuals[:, target_col_idx] = actuals_flat_scaled
        actuals = scaler.inverse_transform(dummy_actuals)[:, target_col_idx]

        dummy_predictions = np.zeros((len(predictions_flat_scaled), num_features))
        dummy_predictions[:, target_col_idx] = predictions_flat_scaled
        predictions = scaler.inverse_transform(dummy_predictions)[:, target_col_idx]

        all_stations_actuals.append(actuals)
        all_stations_predictions.append(predictions)

        # --- Plotting for each station ---
        plt.figure(figsize=(15, 7))
        plt.plot(actuals, label='Actual PV Normalized')
        plt.plot(predictions, label='Predicted PV Normalized', linestyle='--', alpha=0.8)
        plt.title(f'Station {station_idx} - Actual vs. Predicted (Test Set)')
        plt.xlabel('Time Step (hours)')
        plt.ylabel('Normalized PV Production')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"station_{station_idx}_actual_vs_predicted.png")
        plt.show()

    # --- Calculate overall metrics across all stations ---
    overall_actuals = np.concatenate(all_stations_actuals)
    overall_predictions = np.concatenate(all_stations_predictions)
    
    mse = mean_squared_error(overall_actuals, overall_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(overall_actuals, overall_predictions)
    r2 = r2_score(overall_actuals, overall_predictions)

    print("\n--- Overall Evaluation Results (Across All Stations) ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R-squared (R²): {r2:.4f}")


if __name__ == '__main__':
    evaluate_model(EVAL_CONFIG)