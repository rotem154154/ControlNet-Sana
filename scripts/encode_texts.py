# scripts/encode_texts.py
import os
import json
import torch
import h5py
import pandas as pd
from diffusers import SanaPipeline
from tqdm import tqdm

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    variant="fp16",
    torch_dtype=torch.float16,
).to("cuda")

pipe.text_encoder.to(torch.bfloat16)
pipe.text_encoder.eval()
for param in pipe.text_encoder.parameters():
    param.requires_grad = False

default_prompt = "A high-quality, well-lit portrait of a human face, centered, looking at the camera"

csv_path = "/path/to/SFHQ_T2I_dataset.csv"  # Update this path
df = pd.read_csv(csv_path)
num_samples = len(df)
print(f"Found {num_samples} samples.")

with torch.no_grad():
    pos_embed, _, neg_embed, _ = pipe.encode_prompt(
        default_prompt,
        negative_prompt="",
        device="cuda",
        clean_caption=True,
    )
dummy_cond = torch.cat([neg_embed, pos_embed], dim=0)[:, :100, :]
hidden_dim = dummy_cond.shape[-1]
print("Hidden dim:", hidden_dim)

h5_path = "text_encodings.h5"
hf = h5py.File(h5_path, "w")
dset = hf.create_dataset("cond_embeds", shape=(num_samples, 2, 100, hidden_dim), dtype="float32")

for i, row in tqdm(df.iterrows(), total=num_samples):
    try:
        config = json.loads(row["configs"])
        prompt = config.get("orig_prompt", "").strip()
    except Exception:
        prompt = ""
    if not prompt:
        prompt = str(row.get("text_prompt", "")).strip()
    if not prompt:
        prompt = default_prompt
    with torch.no_grad():
        pos_embed, _, neg_embed, _ = pipe.encode_prompt(
            prompt,
            negative_prompt="",
            device="cuda",
            clean_caption=True,
        )
    cond_embed = torch.cat([neg_embed, pos_embed], dim=0)[:, :100, :]
    dset[i, :, :, :] = cond_embed.cpu().numpy()
    if i % 100 == 0:
        torch.cuda.empty_cache()

hf.close()
print(f"Saved text encodings to {h5_path}")
