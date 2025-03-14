# train.py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from data.dataset_features import ControlNetDataModule
from models.finetuner_pruning_lora import ControlNetFineTuner
from utils.helpers import seed_everything
from config.config import Config
import lovely_tensors as lt
lt.monkey_patch()

def train_model():
    base_path = '../sana/'
    seed_everything(Config.seed)
    wandb_logger = WandbLogger(project=Config.wandb_project, log_model=True)
    data_module = ControlNetDataModule(
        image_latents_path=base_path+"t2i_sfhq.h5",
        text_encodings_path=base_path+"text_encodings.h5",
        batch_size=Config.batch_size,
        holdout_ids_file=base_path+"holdout_ids.npy",
        image_encodings_dir='/root/.cache/kagglehub/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models/versions/1/pretrained_features/pretrained_features',
    )

    model = ControlNetFineTuner(load_ckpt=False, cfg_scale=3.0)
    data_module.setup()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=base_path+"fine_tuned_models",
        filename="model_epoch_{epoch:02d}_step_{step}",
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=5000,
    )

    class LrLogger(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            pl_module.log("lr", current_lr, on_step=True, prog_bar=True)

    trainer = pl.Trainer(
        max_epochs=Config.num_epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16",
        default_root_dir=base_path+"fine_tuned_models",
        gradient_clip_val=1.0,
        logger=wandb_logger,
        log_every_n_steps=1,
        accumulate_grad_batches=Config.accumulate_grad_batches,
        callbacks=[checkpoint_callback, LrLogger()]
    )

    trainer.fit(model, datamodule=data_module)
    import wandb
    wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    train_model()
