# scripts/encode_images.py
import os
import h5py
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import SanaPipeline
import torch.nn as nn
import torch.nn.functional as F


class HEDNetwork(nn.Module):
    def __init__(self, model='bsds500'):
        super().__init__()
        # Define VGG-like layers
        self.netVggOne = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )
        self.netVggTwo = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )
        self.netVggThr = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )
        self.netVggFou = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )
        self.netVggFiv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )
        # Side output layers
        self.netScoreOne = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        # Combine side outputs
        self.netCombine = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # Load pre-trained weights (adjust keys if needed)
        state_dict = torch.hub.load_state_dict_from_url(
            url='http://content.sniklaus.com/github/pytorch-hed/network-' + model + '.pytorch',
            file_name='hed-' + model
        )
        state_dict = { key.replace('module', 'net'): value for key, value in state_dict.items() }
        self.load_state_dict(state_dict)

    def forward(self, tenInput):
        # Expect tenInput in [0,1] range with shape (B,3,H,W)
        tenInput = tenInput * 255.0
        # Subtract BGR mean (the model was trained on BGR images)
        mean = torch.tensor([104.00698793, 116.66876762, 122.67891434],
                            dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
        tenInput = tenInput - mean

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        # Upsample side outputs to match input size
        h, w = tenInput.shape[2], tenInput.shape[3]
        tenScoreOne = F.interpolate(tenScoreOne, size=(h, w), mode='bilinear', align_corners=False)
        tenScoreTwo = F.interpolate(tenScoreTwo, size=(h, w), mode='bilinear', align_corners=False)
        tenScoreThr = F.interpolate(tenScoreThr, size=(h, w), mode='bilinear', align_corners=False)
        tenScoreFou = F.interpolate(tenScoreFou, size=(h, w), mode='bilinear', align_corners=False)
        tenScoreFiv = F.interpolate(tenScoreFiv, size=(h, w), mode='bilinear', align_corners=False)

        # Combine the side outputs
        combined = self.netCombine(torch.cat(
            [tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv],
            dim=1
        ))
        return combined

def get_edge_map(pil_img, hed_model, device, target_size=(512, 512)):
    img_resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    np_img = np.array(img_resized, dtype=np.float32) / 255.0
    np_img = np_img[:, :, ::-1]  # Convert RGB to BGR
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np.ascontiguousarray(np_img)
    tensor_img = torch.from_numpy(np_img).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        edge_tensor = hed_model(tensor_img)
    edge_np = edge_tensor.squeeze().cpu().numpy()
    edge_uint8 = (edge_np * 255.0).clip(0, 255).astype(np.uint8)
    edge_pil = Image.fromarray(edge_uint8, mode="L").convert("RGB")
    return edge_pil

def create_hdf5_dataset(image_root, output_file, pipe, hed_model, image_size=512):
    with h5py.File(output_file, 'w') as hf:
        latent_dtype = np.float16
        label_dtype = np.int8

        hf.create_dataset('latents', shape=(0, 32, 16, 16), maxshape=(None, 32, 16, 16), dtype=latent_dtype)
        hf.create_dataset('edge_latents', shape=(0, 32, 16, 16), maxshape=(None, 32, 16, 16), dtype=latent_dtype)
        hf.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=label_dtype)

        image_paths = []
        labels = []
        for root, dirs, files in os.walk(image_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
                    labels.append(1)

        batch_size = 4
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size)):
                batch_paths = image_paths[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]

                batch_orig = []
                batch_edge = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                        orig_proc = pipe.image_processor.preprocess(img_resized, height=image_size, width=image_size)
                        batch_orig.append(orig_proc)
                        edge_img = get_edge_map(img_resized, hed_model, device=pipe.device, target_size=(image_size, image_size))
                        edge_proc = pipe.image_processor.preprocess(edge_img, height=image_size, width=image_size)
                        batch_edge.append(edge_proc)
                    except Exception as e:
                        print(f"Skipping corrupt image {path}: {str(e)}")
                        continue

                if not batch_orig or not batch_edge:
                    continue

                batch_orig_tensor = torch.cat(batch_orig).to(device=pipe.device, dtype=torch.float16)
                batch_edge_tensor = torch.cat(batch_edge).to(device=pipe.device, dtype=torch.float16)

                orig_latents = pipe.vae.encoder(batch_orig_tensor)
                edge_latents = pipe.vae.encoder(batch_edge_tensor)
                orig_latents = orig_latents.cpu().numpy().astype(latent_dtype)
                edge_latents = edge_latents.cpu().numpy().astype(latent_dtype)

                new_size = hf['latents'].shape[0] + orig_latents.shape[0]
                hf['latents'].resize((new_size, 32, 16, 16))
                hf['edge_latents'].resize((new_size, 32, 16, 16))
                hf['labels'].resize((new_size,))

                hf['latents'][-orig_latents.shape[0]:] = orig_latents
                hf['edge_latents'][-edge_latents.shape[0]:] = edge_latents
                hf['labels'][-len(batch_labels):] = np.array(batch_labels, dtype=label_dtype)

if __name__ == "__main__":
    pipe = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_600M_512px_diffusers",
        variant="fp16",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.vae.eval()
    pipe.vae.requires_grad_(False)

    hed_model = HEDNetwork(model='bsds500').to(pipe.device)
    hed_model.eval()
    for p in hed_model.parameters():
        p.requires_grad = False

    create_hdf5_dataset(
        image_root='/path/to/images',  # Update this path
        output_file="t2i_sfhq.h5",
        pipe=pipe,
        hed_model=hed_model,
        image_size=512
    )
