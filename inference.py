# inference.py
import torch
import h5py
from diffusers import SanaPipeline
from PIL import Image
import numpy as np
from models.finetuner import ControlNetFineTuner
import gc

def load_sample_from_h5(image_latents_path, text_encodings_path, sample_idx=0):
    with h5py.File(image_latents_path, 'r') as image_hf:
        edge_latent = torch.from_numpy(image_hf['edge_latents'][sample_idx]).float()
        original_latent = torch.from_numpy(image_hf['latents'][sample_idx]).float()

    with h5py.File(text_encodings_path, 'r') as text_hf:
        text_enc = torch.from_numpy(text_hf['cond_embeds'][sample_idx]).float()
        uncond_text_enc = text_enc[0]
        cond_text_enc = text_enc[1]

    return edge_latent, uncond_text_enc, cond_text_enc, original_latent

def generate_image(model, latents, uncond_embeds, cond_embeds, edge_latent, cfg_scale=4.5, num_inference_steps=50):
    uncond = uncond_embeds.unsqueeze(0)
    cond = cond_embeds.unsqueeze(0)
    encoder_hidden_states = torch.cat([uncond, cond], dim=0)

    edge_latent = (edge_latent * model.pipe.vae.config.scaling_factor)
    edge_latent_gen = model.transformer.patch_embed(edge_latent.unsqueeze(0))

    model.pipe.scheduler.set_timesteps(num_inference_steps)
    with torch.no_grad():
        for t in model.pipe.scheduler.timesteps:
            model_input = torch.cat([latents, latents], dim=0)
            t_tensor = t.expand(model_input.shape[0]).to(latents.device)
            noise_pred = model.transformer(
                model_input,
                edge_latent_gen,
                encoder_hidden_states=encoder_hidden_states,
                timestep=t_tensor,
                return_dict=False
            )[0]
            noise_uncond, noise_text = noise_pred.chunk(2)
            guided_noise = noise_uncond + cfg_scale * (noise_text - noise_uncond)
            latents = model.pipe.scheduler.step(guided_noise, t, latents, return_dict=False)[0]

    latents_scaled = latents / model.pipe.vae.config.scaling_factor
    decoded = model.pipe.vae.decode(latents_scaled)
    image = model.pipe.image_processor.postprocess(decoded.sample, output_type="pil")[0]
    model.pipe.scheduler.set_timesteps(1000)
    return image

def run_inference(sample_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControlNetFineTuner(load_ckpt=True, cfg_scale=3.0)
    model = model.to(device)
    model.eval()
    model.pipe.vae.to(torch.bfloat16)
    model.pipe.vae.eval()

    edge_latent, uncond_text_enc, cond_text_enc, _ = load_sample_from_h5("t2i_sfhq.h5", "text_encodings.h5", sample_idx)
    latent_shape = (1, model.transformer.config.in_channels, 512 // 32, 512 // 32)
    latents = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)

    generated_image = generate_image(model, latents, uncond_text_enc, cond_text_enc, edge_latent, cfg_scale=4.5, num_inference_steps=20)
    generated_image.save(f"generated_sample_{sample_idx}.png")
    print(f"Generated image saved as generated_sample_{sample_idx}.png")

if __name__ == "__main__":
    run_inference()
