# config/config.py
class Config:
    # Training hyperparameters
    batch_size = 6
    num_epochs = 20
    learning_rate = 1e-5
    weight_decay = 1e-2
    accumulate_grad_batches = 16 # high for stable sudden convergence

    # WandB and project settings
    wandb_project = "ControlNet-Sana"

    # Other parameters
    seed = 1
