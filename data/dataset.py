# data/dataset.py
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl

def worker_init_fn(worker_id, seed=1):
    np.random.seed(seed + worker_id)

class HDF5LatentEdgeTextEncodingDataset(Dataset):
    """
    Loads precomputed image latents, edge latents, and text encodings from HDF5 files.
    """
    def __init__(self, image_latents_path, text_encodings_path):
        self.image_latents_path = image_latents_path
        self.text_encodings_path = text_encodings_path
        self.image_hf = None
        self.text_hf = None

        with h5py.File(self.image_latents_path, 'r') as hf:
            self.num_samples = hf['latents'].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.image_hf is None:
            self.image_hf = h5py.File(self.image_latents_path, 'r')
        if self.text_hf is None:
            self.text_hf = h5py.File(self.text_encodings_path, 'r')

        latent = torch.from_numpy(self.image_hf['latents'][idx]).float()
        edge_latent = torch.from_numpy(self.image_hf['edge_latents'][idx]).float()
        text_enc = torch.from_numpy(self.text_hf['cond_embeds'][idx]).float()

        return latent, edge_latent, text_enc

class ControlNetDataModule(pl.LightningDataModule):
    """
    DataModule for handling HDF5 datasets used in ControlNet training.
    """
    def __init__(self, image_latents_path, text_encodings_path, batch_size, holdout_ids_file):
        super().__init__()
        self.image_latents_path = image_latents_path
        self.text_encodings_path = text_encodings_path
        self.batch_size = batch_size
        self.holdout_ids_file = holdout_ids_file

    def setup(self, stage=None):
        self.train_dataset = HDF5LatentEdgeTextEncodingDataset(
            self.image_latents_path, self.text_encodings_path
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed=1)
        )
