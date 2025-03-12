# config/config.py
class Config:
    # Training hyperparameters
    batch_size = 8
    num_epochs = 20
    learning_rate = 5e-5
    weight_decay = 1e-2
    accumulate_grad_batches = 4 # high for stable sudden convergence

    # WandB and project settings
    wandb_project = "ControlNet-Sana"

    # Other parameters
    seed = 1
