import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import wandb
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import argparse # To accept arguments from the sweep

# --- CONFIGURATION (Default values, will be overridden by sweep) ---
# This dictionary now acts as a set of default hyperparameters.
CONFIG = {
    "pretrain_data_prefix": "data/transfer_learning_data/",
    "pv_files": ["PV Plants Datasets_62030198.csv", "PV Plants Datasets_62032213.csv", "PV Plants Datasets_73060645.csv", "PV Plants Datasets_73061935.csv", "PV Plants Datasets_84071566.csv", "PV Plants Datasets_84071567.csv", "PV Plants Datasets_84071568.csv", "PV Plants Datasets_84071569.csv", "PV Plants Datasets_84071570.csv"],
    "weather_files": ["weather_files/Braga_weather.csv", "weather_files/Lisbon_weather.csv", "weather_files/Tavira_weather.csv", "weather_files/Loule_weather.csv", "weather_files/Faro_weather.csv", "weather_files/Lisbon_weather.csv", "weather_files/Setubal_weather.csv", "weather_files/Lisbon_weather.csv", "weather_files/Lisbon_weather.csv"],
    "pv_peak_power_kwp": [22_400 for _ in range(9)],
    "validation_station_idx": -1, # Use the last station for validation by default
    "validation_train_hours": 1000, # Hours of the validation station's data to train on
    "input_seq_len": 48,
    "output_seq_len": 24,
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "epochs": 100, # Set a high max, early stopping will handle it
    "batch_size": 256,
    "learning_rate": 0.0001,
    "early_stopping_patience": 10, # Stop after 10 epochs with no improvement
}

# --- MODEL & DATA HANDLING (No changes from previous version) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.pe: torch.Tensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(0)])

class PVTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, 1)
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.input_embedding(src) * np.sqrt(self.d_model)
        tgt = self.input_embedding(tgt) * np.sqrt(self.d_model)
        src = self.pos_encoder(src.permute(1, 0, 2)).permute(1, 0, 2)
        tgt = self.pos_encoder(tgt.permute(1, 0, 2)).permute(1, 0, 2)
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output)

class TimeseriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

def create_sequences(data, input_len, output_len):
    xs, ys = [], []
    for i in range(len(data) - input_len - output_len + 1):
        xs.append(data[i : (i + input_len)])
        ys.append(data[(i + input_len - 1) : (i + input_len + output_len - 1)])
    return np.array(xs), np.array(ys)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- REIMPLEMENTED DATA PROCESSING FOR DEDICATED VALIDATION ---
def load_and_prepare_data(config):
    station_dfs = []
    for pv_file, weather_file, peak_power in zip(config["pv_files"], config["weather_files"], config["pv_peak_power_kwp"]):
        # ... (same loading logic as before) ...
        df_pv = pd.read_csv(os.path.join(config["pretrain_data_prefix"], pv_file))
        df_pv['Date'] = pd.to_datetime(df_pv['Date'])
        if 'Specific Energy (kWh/kWp)' in df_pv.columns:
            df_pv['pv_normalized'] = df_pv['Specific Energy (kWh/kWp)']
        elif 'Power Output (Watts)' in df_pv.columns:
            df_pv['pv_normalized'] = (df_pv['Power Output (Watts)'] / 1000) / peak_power
        else:
            raise ValueError(f"No usable production column in {pv_file}")
        df_weather = pd.read_csv(os.path.join(config["pretrain_data_prefix"], weather_file), parse_dates=['time'])
        df_merged = pd.merge(df_pv[['Date', 'pv_normalized']], df_weather, left_on='Date', right_on='time', how='inner')
        station_dfs.append(df_merged.set_index('Date').sort_index())

    # Separate the validation station's dataframe
    val_station_df = station_dfs.pop(config["validation_station_idx"])
    train_station_dfs = station_dfs

    # --- Feature Engineering and Global Scaler Fitting ---
    master_df_for_scaling = pd.concat(train_station_dfs + [val_station_df]) # Use all data to fit scaler
    # ... (same feature engineering as before) ...
    master_df_for_scaling['hour'] = master_df_for_scaling.index.hour
    master_df_for_scaling['month'] = master_df_for_scaling.index.month
    master_df_for_scaling['hour_sin'] = np.sin(2 * np.pi * master_df_for_scaling['hour'] / 23.0)
    master_df_for_scaling['hour_cos'] = np.cos(2 * np.pi * master_df_for_scaling['hour'] / 23.0)
    month_dummies = pd.get_dummies(master_df_for_scaling['month'], prefix='month').reindex(columns=[f'month_{i}' for i in range(1, 13)], fill_value=0)
    master_df_for_scaling = pd.concat([master_df_for_scaling, month_dummies], axis=1)
    feature_cols = ['pv_normalized', 'temperature_2m (°C)', 'relative_humidity_2m (%)', 
                    'shortwave_radiation (W/m²)', 'hour_sin', 'hour_cos'] + list(month_dummies.columns)
    final_df_for_scaling = master_df_for_scaling[feature_cols].copy().rename(columns={
        'temperature_2m (°C)': 'temp', 'relative_humidity_2m (%)': 'humidity', 'shortwave_radiation (W/m²)': 'radiation'
    })

    scaler = StandardScaler().fit(final_df_for_scaling.values)

    # --- Process training stations ---
    training_data_list = []
    for df in train_station_dfs:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)
        month_dummies = pd.get_dummies(df['month'], prefix='month').reindex(columns=[f'month_{i}' for i in range(1, 13)], fill_value=0)
        df = pd.concat([df, month_dummies], axis=1)
        final_df_station = df[feature_cols].copy().rename(columns={'temperature_2m (°C)': 'temp', 'relative_humidity_2m (%)': 'humidity', 'shortwave_radiation (W/m²)': 'radiation'})
        training_data_list.append(scaler.transform(final_df_station.values))
        
    # --- Process the dedicated validation station ---
    val_station_df['hour'] = val_station_df.index.hour
    val_station_df['month'] = val_station_df.index.month
    val_station_df['hour_sin'] = np.sin(2 * np.pi * val_station_df['hour'] / 23.0)
    val_station_df['hour_cos'] = np.cos(2 * np.pi * val_station_df['hour'] / 23.0)
    month_dummies = pd.get_dummies(val_station_df['month'], prefix='month').reindex(columns=[f'month_{i}' for i in range(1, 13)], fill_value=0)
    val_station_df = pd.concat([val_station_df, month_dummies], axis=1)
    final_val_station_df = val_station_df[feature_cols].copy().rename(columns={'temperature_2m (°C)': 'temp', 'relative_humidity_2m (%)': 'humidity', 'shortwave_radiation (W/m²)': 'radiation'})
    
    # Split the validation station's data: a small part for training, the rest for validation
    val_train_part = final_val_station_df.iloc[:config["validation_train_hours"]].values
    val_test_part = final_val_station_df.iloc[config["validation_train_hours"]:].values
    
    training_data_list.append(scaler.transform(val_train_part)) # Add small part to training pool
    validation_data_scaled = scaler.transform(val_test_part) # This is our hold-out set
    
    target_col_idx = final_val_station_df.columns.get_loc('pv_normalized')

    print(f"Data processed. {len(training_data_list)} datasets for training. 1 dataset for validation.")
    return training_data_list, validation_data_scaled, scaler, target_col_idx, final_val_station_df.columns


# --- REIMPLEMENTED TRAINING LOOP WITH EARLY STOPPING ---
def train(config):
    wandb.init(project="PV Load Pred", config=config)
    config = wandb.config # Get the config from wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_data_list, validation_data, scaler, target_col_idx, feature_names = load_and_prepare_data(config)
    
    num_features = len(feature_names)
    model = PVTransformer(num_features=num_features, d_model=config.d_model, nhead=config.nhead,
                          num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers,
                          dim_feedforward=config.dim_feedforward, dropout=config.dropout).to(device)
    
    print(f"Model initialized with {count_parameters(model):,} parameters.")
    wandb.log({"parameters": count_parameters(model)})

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    best_val_r2 = -float('inf')
    epochs_no_improve = 0

    # Prepare validation loader (created once)
    X_val, y_val = create_sequences(validation_data, config.input_seq_len, config.output_seq_len)
    val_dataset = TimeseriesDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        total_epoch_train_loss = 0
        for train_data_scaled in training_data_list:
            X_train, y_train = create_sequences(train_data_scaled, config.input_seq_len, config.output_seq_len)
            train_dataset = TimeseriesDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                prediction = model(src, tgt[:, :-1, :])
                loss = criterion(prediction, tgt[:, 1:, target_col_idx].unsqueeze(-1))
                loss.backward()
                optimizer.step()
                total_epoch_train_loss += loss.item()
        
        avg_epoch_train_loss = total_epoch_train_loss / len(training_data_list)

        # --- Validation on the dedicated station ---
        model.eval()
        val_loss = 0
        all_preds_scaled, all_actuals_scaled = [], []
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                prediction = model(src, tgt[:, :-1, :])
                loss = criterion(prediction, tgt[:, 1:, target_col_idx].unsqueeze(-1))
                val_loss += loss.item()
                all_preds_scaled.append(prediction.cpu().numpy())
                all_actuals_scaled.append(tgt[:, 1:, target_col_idx].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)

        # Calculate R² score for validation
        preds_flat = np.concatenate([p.flatten() for p in all_preds_scaled])
        actuals_flat = np.concatenate([a.flatten() for a in all_actuals_scaled])
        val_r2 = r2_score(actuals_flat, preds_flat)
        wandb.log({"epoch": epoch, "train_loss": avg_epoch_train_loss, "validation_loss": avg_val_loss, "validation_r2": val_r2})
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_epoch_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val R²: {val_r2:.4f}")

        # --- Early Stopping Logic ---
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            epochs_no_improve = 0
            os.makedirs("checkpoints", exist_ok=True)
            # Save the model associated with this specific run
            model_path = os.path.join(wandb.run.dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with R²: {best_val_r2:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break
            
    wandb.finish()

if __name__ == '__main__':
    # --- Argument Parsing for Sweep ---
    parser = argparse.ArgumentParser()
    # Add arguments for all hyperparameters you want to sweep
    parser.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'])
    parser.add_argument('--d_model', type=int, default=CONFIG['d_model'])
    parser.add_argument('--dim_feedforward', type=int, default=CONFIG['dim_feedforward'])
    parser.add_argument('--num_encoder_layers', type=int, default=CONFIG['num_encoder_layers'])
    parser.add_argument('--num_decoder_layers', type=int, default=CONFIG['num_decoder_layers'])
    parser.add_argument('--dropout', type=float, default=CONFIG['dropout'])
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    args = parser.parse_args()

    # Update CONFIG with parsed arguments
    config_run = CONFIG.copy()
    config_run.update(vars(args))

    train(config_run)