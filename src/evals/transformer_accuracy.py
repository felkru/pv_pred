import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import random # For setting seeds

# --- Import Model and Data Loading Logic from your pretrain_script.py ---
# For simplicity, I'm copying them here, but in a real project, you'd
# import these from your shared model/data processing modules.
from ..transformer_based.pretrain import PVTransformer, TimeseriesDataset, create_sequences, load_and_process_data, CONFIG as PRETRAIN_CONFIG

# --- CONFIGURATION FOR EVALUATION ---
# Use relevant parts of the pretrain config.
# Ensure these match the config used during pre-training.
EVAL_CONFIG = {
    "pretrain_data_prefix": PRETRAIN_CONFIG["pretrain_data_prefix"],
    "pv_files": PRETRAIN_CONFIG["pv_files"],
    "weather_files": PRETRAIN_CONFIG["weather_files"],
    "pv_peak_power_kwp": PRETRAIN_CONFIG["pv_peak_power_kwp"],
    "train_split": PRETRAIN_CONFIG["train_split"], # Important to get the same test set split
    "input_seq_len": PRETRAIN_CONFIG["input_seq_len"],
    "output_seq_len": PRETRAIN_CONFIG["output_seq_len"],
    "d_model": PRETRAIN_CONFIG["d_model"],
    "nhead": PRETRAIN_CONFIG["nhead"],
    "num_encoder_layers": PRETRAIN_CONFIG["num_encoder_layers"],
    "num_decoder_layers": PRETRAIN_CONFIG["num_decoder_layers"],
    "dim_feedforward": PRETRAIN_CONFIG["dim_feedforward"],
    "dropout": PRETRAIN_CONFIG["dropout"],
    "batch_size": PRETRAIN_CONFIG["batch_size"], # Use same batch size for consistency
    "model_path": "checkpoints/devout_wildflower/best.pth", # Path to your saved model
    "random_seed": 42, # Consistent seed for reproducibility
}

def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def evaluate_model(config):
    set_seed(config["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # Load and process data, getting the SAME test set and scaler used during training
    _, test_data_scaled, scaler, target_col_idx, feature_names = load_and_process_data(config)
    
    # Create sequences for the test set
    X_test, y_test_decoder_input = create_sequences(test_data_scaled, config["input_seq_len"], config["output_seq_len"])
    test_dataset = TimeseriesDataset(X_test, y_test_decoder_input)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize model with the SAME architecture parameters
    num_features = test_data_scaled.shape[1]
    model = PVTransformer(
        num_features=num_features, d_model=config["d_model"], nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"], num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"]
    ).to(device)

    # Load the state dictionary from the best pre-trained model
    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"Model file not found at: {config['model_path']}. "
                                "Make sure you run pretrain_script.py first and it saves the model.")
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval() # Set model to evaluation mode

    all_predictions = []
    all_actuals = []

    print("Starting evaluation...")
    with torch.no_grad():
        for src, tgt_input in test_loader:
            src, tgt_input = src.to(device), tgt_input.to(device)
            # The actual target values for comparison are the PV production for the next `output_seq_len` steps
            # from the `tgt_input` sequence, starting from the second element (shifted by 1).
            batch_actuals_scaled = tgt_input[:, 1:, target_col_idx].cpu().numpy().squeeze()
            
            # Predict
            # The decoder input for prediction should be just the first element of the target sequence
            # for the first prediction, then autoregressively use previous predictions.
            # However, for evaluation with teacher forcing, we use the true sequence shifted:
            batch_predictions_scaled = model(src, tgt_input[:, :-1, :]).cpu().numpy().squeeze()
            
            # Reshape if output_seq_len is 1 (squeeze might remove the dim)
            if config["output_seq_len"] == 1:
                batch_actuals_scaled = batch_actuals_scaled.reshape(-1, 1)
                batch_predictions_scaled = batch_predictions_scaled.reshape(-1, 1)

            all_predictions.append(batch_predictions_scaled)
            all_actuals.append(batch_actuals_scaled)

    # Concatenate all predictions and actuals
    # Note: Flattening is needed if output_seq_len > 1 and you want a single list of all predictions/actuals
    # for metric calculation across the entire test set.
    actuals_flat_scaled = np.concatenate(all_actuals).flatten()
    predictions_flat_scaled = np.concatenate(all_predictions).flatten()

    # Inverse transform to get back to original scale (important for interpretable metrics)
    # Create dummy arrays to inverse transform only the 'pv_normalized' column
    # The scaler was fitted on ALL features. So, we need to create a dummy array
    # with the same number of features and place our scaled values in the target column's position.
    dummy_actuals = np.zeros((actuals_flat_scaled.shape[0], num_features))
    dummy_actuals[:, target_col_idx] = actuals_flat_scaled
    actuals = scaler.inverse_transform(dummy_actuals)[:, target_col_idx]

    dummy_predictions = np.zeros((predictions_flat_scaled.shape[0], num_features))
    dummy_predictions[:, target_col_idx] = predictions_flat_scaled
    predictions = scaler.inverse_transform(dummy_predictions)[:, target_col_idx]
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("\n--- Evaluation Results ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R-squared (R²): {r2:.4f}")

    # Plotting (Actual vs. Predicted)
    plt.figure(figsize=(12, 6))
    # plt.plot(actuals[:200], label='Actual PV Normalized')
    # plt.plot(predictions[:200], label='Predicted PV Normalized', linestyle='--')
    plt.plot(actuals[20500:50000], label='Actual PV Normalized')
    plt.plot(predictions[20500:50000], label='Predicted PV Normalized', linestyle='--')
    plt.title('Actual vs. Predicted PV Production (First 200 hours of Test Set)')
    plt.xlabel('Time Step (hours)')
    plt.ylabel('Normalized PV Production')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("actual_vs_predicted_pretrain.png")
    plt.show()

    # Plotting (Residuals)
    residuals = predictions - actuals
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title('Distribution of Prediction Residuals')
    plt.xlabel('Residual (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("residuals_distribution_pretrain.png")
    plt.show()


if __name__ == '__main__':
    # Ensure data directory exists (user is expected to populate it)
    if not os.path.exists(EVAL_CONFIG["pretrain_data_prefix"]):
        print(f"Error: Data directory '{EVAL_CONFIG['pretrain_data_prefix']}' not found.")
        print("Please create this directory and place your PV and weather CSV files inside.")
        exit()

    evaluate_model(EVAL_CONFIG)