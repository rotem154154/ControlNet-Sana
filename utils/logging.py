# utils/logging.py
import wandb

def init_wandb(project_name):
    wandb.init(project=project_name, log_model=True)
