import wandb
import yaml
import os

# --- 1. Define the Sweep Configuration ---
# The command is now explicitly defined to use the python interpreter.
sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'validation_r2',
      'goal': 'maximize'
    },
    # The 'program' key tells the agent which file to run.
    'program': 'src/transformer_based/pretrain.py',
    
    # The 'command' key tells the agent HOW to run it.
    'command': [
        'python',         # <-- FIX: Explicitly use the 'python' interpreter
        '${program}',     # This macro inserts the file path from the 'program' key
        '${args}'         # This macro injects all the hyperparameter arguments
    ],
    'parameters': {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.00001,
            'max': 0.001
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.3
        },
        'd_model': {
            'values': [64, 128, 256]
        },
        'dim_feedforward': {
            'values': [256, 512, 1024]
        },
        'num_encoder_layers': {
            'values': [2, 3, 4]
        },
        'num_decoder_layers': {
            'values': [2, 3, 4]
        },
        'batch_size': {
            'values': [128, 256]
        }
    }
}

# --- 2. Initialize the Sweep ---
def initialize_sweep():
    with open('sweep_config.yaml', 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)

    print("--- Sweep Configuration ---")
    print(yaml.dump(sweep_config))
    print("--------------------------")

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="PV Load Pred",
        entity="felkru-rwth-aachen-university"
    )
    
    print(f"\nSweep initialized. View it at: https://wandb.ai/felkru-rwth-aachen-university/PV%20Load%20Pred/sweeps/{sweep_id}")
    
    full_agent_command = f'wandb agent felkru-rwth-aachen-university/"PV Load Pred"/{sweep_id}'
    
    print("\n--- To run the agent, copy and paste the command below into your terminal ---")
    print(full_agent_command)
    print("-----------------------------------------------------------------------------")

if __name__ == '__main__':
    initialize_sweep()