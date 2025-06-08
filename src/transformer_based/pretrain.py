import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import wandb
import os
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
# NOTE: Your configuration is kept as is. It looks good.
CONFIG = {
    "pretrain_data_prefix": "data/transfer_learning_data/",
    "pv_files": [
        "PV Plants Datasets_62030198.csv",
        "PV Plants Datasets_62032213.csv",
    ],
    "weather_files": [
        "weather_files/Braga_weather.csv",
        "weather_files/Lisbon_weather.csv",
    ],
    "pv_peak_power_kwp": [22_400 for _ in range(2)],
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


# --- MODEL DEFINITION ---

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
        # FIX for linter: Explicitly declare 'pe' as a Tensor.
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be of shape [seq_len, batch_size, d_model]
        return self.dropout(x + self.pe[:x.size(0)])

class PVTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, 1) # Predicts normalized PV production

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # FIX for model logic: Correctly handle permutations for batch_first Transformer
        # src shape: [batch, src_seq_len, features]
        # tgt shape: [batch, tgt_seq_len, features]

        # 1. Embed and scale
        src = self.input_embedding(src) * np.sqrt(self.d_model)
        tgt = self.input_embedding(tgt) * np.sqrt(self.d_model)

        # 2. Permute from [batch, seq, feat] to [seq, batch, feat] for PositionalEncoding
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # 3. Apply positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # 4. Permute back to [batch, seq, feat] for batch_first Transformer
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # 5. Generate masks (no change needed here)
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # 6. Pass through transformer and final layer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output)


# --- DATA HANDLING ---

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

def load_and_process_data(config):
    all_dfs = []
    for pv_file, weather_file, peak_power in zip(config["pv_files"], config["weather_files"], config["pv_peak_power_kwp"]):
        df_pv = pd.read_csv(os.path.join(config["pretrain_data_prefix"], pv_file))
        # FIX for runtime error: Removed dayfirst=True to allow pandas to auto-detect MM/DD/YYYY format.
        df_pv['Date'] = pd.to_datetime(df_pv['Date'])

        if 'Specific Energy (kWh/kWp)' in df_pv.columns:
            df_pv['pv_normalized'] = df_pv['Specific Energy (kWh/kWp)']
        elif 'Produced Energy (kWh)' in df_pv.columns:
            df_pv['pv_normalized'] = df_pv['Produced Energy (kWh)'] / peak_power
        elif 'Power Output (Watts)' in df_pv.columns:
            df_pv['pv_normalized'] = (df_pv['Power Output (Watts)'] / 1000) / peak_power
        else:
            raise ValueError(f"No usable production column in {pv_file}")
        
        df_weather = pd.read_csv(os.path.join(config["pretrain_data_prefix"], weather_file), parse_dates=['time'])
        df_merged = pd.merge(df_pv[['Date', 'pv_normalized']], df_weather, left_on='Date', right_on='time', how='inner')
        all_dfs.append(df_merged)

    full_df = pd.concat(all_dfs, ignore_index=True).set_index('Date').sort_index()

    full_df['hour'] = full_df.index.hour
    full_df['month'] = full_df.index.month
    full_df['hour_sin'] = np.sin(2 * np.pi * full_df['hour'] / 23.0)
    full_df['hour_cos'] = np.cos(2 * np.pi * full_df['hour'] / 23.0)
    
    month_dummies = pd.get_dummies(full_df['month'], prefix='month').reindex(columns=[f'month_{i}' for i in range(1, 13)], fill_value=0)
    full_df = pd.concat([full_df, month_dummies], axis=1)

    feature_cols = [
        'pv_normalized', 'temperature_2m (°C)', 'relative_humidity_2m (%)', 
        'shortwave_radiation (W/m²)', 'hour_sin', 'hour_cos'
    ] + list(month_dummies.columns)
    
    final_df = full_df[feature_cols].copy().rename(columns={
        'temperature_2m (°C)': 'temp',
        'relative_humidity_2m (%)': 'humidity',
        'shortwave_radiation (W/m²)': 'radiation'
    })
    
    TARGET_COL = 'pv_normalized'
    target_col_idx = final_df.columns.get_loc(TARGET_COL)
    
    split_idx = int(len(final_df) * config["train_split"])
    train_data = final_df.iloc[:split_idx].values
    test_data = final_df.iloc[split_idx:].values

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    print(f"Data processed. Total shape: {final_df.shape}. Number of features: {len(final_df.columns)}")
    return train_scaled, test_scaled, scaler, target_col_idx, final_df.columns

def train(config):
    wandb.init(project="PV Load Pred", entity="felkru-rwth-aachen-university", config=config)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data_scaled, test_data_scaled, scaler, target_col_idx, _ = load_and_process_data(config)
    
    X_train, y_train_decoder_input = create_sequences(train_data_scaled, config.input_seq_len, config.output_seq_len)
    X_test, y_test_decoder_input = create_sequences(test_data_scaled, config.input_seq_len, config.output_seq_len)

    train_dataset = TimeseriesDataset(X_train, y_train_decoder_input)
    test_dataset = TimeseriesDataset(X_test, y_test_decoder_input)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    num_features = train_data_scaled.shape[1]
    model = PVTransformer(
        num_features=num_features, d_model=config.d_model, nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward, dropout=config.dropout
    ).to(device)

    wandb.watch(model, log='all', log_freq=100)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        for src, tgt_input in train_loader:
            src, tgt_input = src.to(device), tgt_input.to(device)
            tgt_for_pred = tgt_input[:, :-1, :]
            tgt_for_loss = tgt_input[:, 1:, target_col_idx].unsqueeze(-1)

            optimizer.zero_grad()
            prediction = model(src, tgt_for_pred)
            
            loss = criterion(prediction, tgt_for_loss)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt_input in test_loader:
                src, tgt_input = src.to(device), tgt_input.to(device)
                tgt_for_pred = tgt_input[:, :-1, :]
                tgt_for_loss = tgt_input[:, 1:, target_col_idx].unsqueeze(-1)
                
                prediction = model(src, tgt_for_pred)
                val_loss = criterion(prediction, tgt_for_loss)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(test_loader)

        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "validation_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = "checkpoints/pretrained_model_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path}")
            artifact = wandb.Artifact('pv-transformer-pretrained', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    wandb.finish()
    print("Pre-training finished.")

if __name__ == '__main__':
    if not os.path.exists(CONFIG["pretrain_data_prefix"]):
        print(f"Error: Data directory '{CONFIG['pretrain_data_prefix']}' not found.")
        print("Please create this directory and place your PV and weather CSV files inside.")
        exit()

    train(CONFIG)