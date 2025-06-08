import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import wandb
import os
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION (Remains the same) ---
CONFIG = {
    "pretrain_data_prefix": "data/transfer_learning_data/",
    "pv_files": 
        [
            "PV Plants Datasets_62030198.csv",
            "PV Plants Datasets_62032213.csv",
            "PV Plants Datasets_73060645.csv",
            "PV Plants Datasets_73061935.csv",
            "PV Plants Datasets_84071566.csv",
            "PV Plants Datasets_84071567.csv",
            "PV Plants Datasets_84071568.csv",
            "PV Plants Datasets_84071569.csv",
            "PV Plants Datasets_84071570.csv",
        ],
    "weather_files": 
        [
            "weather_files/Braga_weather.csv", 
            "weather_files/Lisbon_weather.csv",
            "weather_files/Tavira_weather.csv",
            "weather_files/Loule_weather.csv",
            "weather_files/Faro_weather.csv",
            "weather_files/Lisbon_weather.csv",
            "weather_files/Setubal_weather.csv",
            "weather_files/Lisbon_weather.csv",
            "weather_files/Lisbon_weather.csv",
        ],
    "pv_peak_power_kwp": [22_400 for _ in range(9)],
    "train_split": 2/3,
    "input_seq_len": 48,
    "output_seq_len": 24,
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.0001,
}

# --- MODEL DEFINITION (No changes needed) ---
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

# --- DATA HANDLING (No changes needed) ---
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

# --- REIMPLEMENTED DATA PROCESSING ---
def load_and_process_data_per_station(config):
    """
    Processes data for each station separately but uses a global scaler.
    Returns:
        - A list of tuples, where each tuple contains (train_scaled, test_scaled, target_col_idx) for a station.
        - The globally fitted scaler object.
        - The feature names.
    """
    station_dfs = []
    for pv_file, weather_file, peak_power in zip(config["pv_files"], config["weather_files"], config["pv_peak_power_kwp"]):
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

    # --- Feature Engineering and Global Scaler Fitting ---
    master_df_for_scaling = pd.concat(station_dfs) # Concatenate temporarily ONLY to fit the scaler

    # Feature engineering on the master dataframe
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
    
    # Fit a single, global scaler on the training part of the combined data
    split_idx_master = int(len(final_df_for_scaling) * config["train_split"])
    scaler = StandardScaler().fit(final_df_for_scaling.iloc[:split_idx_master].values)
    
    # --- Process Each Station Individually using the Global Scaler ---
    processed_stations = []
    for df in station_dfs:
        # Apply same feature engineering
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)
        month_dummies = pd.get_dummies(df['month'], prefix='month').reindex(columns=[f'month_{i}' for i in range(1, 13)], fill_value=0)
        df = pd.concat([df, month_dummies], axis=1)

        final_df_station = df[feature_cols].copy().rename(columns={
            'temperature_2m (°C)': 'temp', 'relative_humidity_2m (%)': 'humidity', 'shortwave_radiation (W/m²)': 'radiation'
        })
        
        # Split and scale using the GLOBAL scaler
        split_idx_station = int(len(final_df_station) * config["train_split"])
        train_data = final_df_station.iloc[:split_idx_station].values
        test_data = final_df_station.iloc[split_idx_station:].values
        
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        target_col_idx = final_df_station.columns.get_loc('pv_normalized')
        processed_stations.append((train_scaled, test_scaled, target_col_idx))

    print(f"Data processed for {len(station_dfs)} stations. Using a global scaler.")
    return processed_stations, scaler, final_df_for_scaling.columns


# --- REIMPLEMENTED TRAINING LOOP ---
def train(config):
    wandb.init(project="PV Load Pred", entity="felkru-rwth-aachen-university", config=config)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processed_stations, _, feature_names = load_and_process_data_per_station(config)
    num_features = len(feature_names)
    
    model = PVTransformer(num_features=num_features, d_model=config.d_model, nhead=config.nhead,
                          num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers,
                          dim_feedforward=config.dim_feedforward, dropout=config.dropout).to(device)

    wandb.watch(model, log='all', log_freq=100)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_val_loss = float('inf')

    print("Starting iterative training across stations...")
    for epoch in range(config.epochs):
        model.train()
        total_epoch_train_loss = 0

        # --- Inner loop for iterating through stations ---
        for station_idx, (train_scaled, _, target_col_idx) in enumerate(processed_stations):
            X_train, y_train = create_sequences(train_scaled, config.input_seq_len, config.output_seq_len)
            train_dataset = TimeseriesDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            
            station_train_loss = 0
            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_for_pred = tgt[:, :-1, :]
                tgt_for_loss = tgt[:, 1:, target_col_idx].unsqueeze(-1)

                optimizer.zero_grad()
                prediction = model(src, tgt_for_pred)
                loss = criterion(prediction, tgt_for_loss)
                loss.backward()
                optimizer.step()
                station_train_loss += loss.item()
            
            avg_station_train_loss = station_train_loss / len(train_loader)
            total_epoch_train_loss += avg_station_train_loss
        
        avg_epoch_train_loss = total_epoch_train_loss / len(processed_stations)

        # --- Validation loop across all stations ---
        model.eval()
        total_epoch_val_loss = 0
        with torch.no_grad():
            for _, (_, test_scaled, target_col_idx) in enumerate(processed_stations):
                X_test, y_test = create_sequences(test_scaled, config.input_seq_len, config.output_seq_len)
                test_dataset = TimeseriesDataset(X_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
                
                station_val_loss = 0
                for src, tgt in test_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    tgt_for_pred = tgt[:, :-1, :]
                    tgt_for_loss = tgt[:, 1:, target_col_idx].unsqueeze(-1)
                    prediction = model(src, tgt_for_pred)
                    station_val_loss += criterion(prediction, tgt_for_loss).item()
                
                total_epoch_val_loss += station_val_loss / len(test_loader)
        
        avg_epoch_val_loss = total_epoch_val_loss / len(processed_stations)

        print(f"Epoch {epoch+1}/{config.epochs} | Avg Train Loss: {avg_epoch_train_loss:.6f} | Avg Val Loss: {avg_epoch_val_loss:.6f}")
        wandb.log({"epoch": epoch, "avg_train_loss": avg_epoch_train_loss, "avg_validation_loss": avg_epoch_val_loss})

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            model_path = "checkpoints/pretrained_model_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with val_loss: {best_val_loss:.6f}")
            artifact = wandb.Artifact('pv-transformer-pretrained', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    wandb.finish()
    print("Pre-training finished.")

if __name__ == '__main__':
    train(CONFIG)