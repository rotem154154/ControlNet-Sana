# models/finetuner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import gc
import pytorch_lightning as pl
from diffusers import SanaPipeline, DPMSolverMultistepScheduler
from typing import Optional, Dict, Any
import wandb
import copy
from peft import LoraConfig, get_peft_model

# Minimal output wrapper; adjust as needed if your pipeline provides its own.
class Transformer2DModelOutput:
    def __init__(self, sample):
        self.sample = sample


def compute_loss_weighting(sigmas, scheme='reciprocal'):
    if scheme == 'none':
        return torch.ones_like(sigmas)
    elif scheme == 'reciprocal':
        return 1.0 / (sigmas ** 2 + 1e-8)  # Avoid division by zero
    elif scheme == 'sigma_sqrt':
        return torch.sqrt(sigmas)
    else:
        return torch.ones_like(sigmas)

class ControlNetFineTuner(pl.LightningModule):
    def __init__(self, load_ckpt, cfg_scale=3.0, text_prompt_ratio=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.text_prompt_ratio = text_prompt_ratio

        # Load the pretrained pipeline in float32.
        self.pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_600M_512px_diffusers",
            torch_dtype=torch.float32,
        ).to("cuda")

        # Convert VAE, transformer, and text encoder to bfloat16.
        self.pipe.vae.to(torch.bfloat16)
        self.pipe.transformer.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)

        # Use Identity for the VAE encoder since we use precomputed latents.
        self.pipe.vae.encoder = nn.Identity()

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "Efficient-Large-Model/Sana_600M_512px_diffusers", subfolder="scheduler"
        )
        self.pipe.scheduler.set_timesteps(1000)

        # Fine-tune only the transformer.
        self.transformer = self.pipe.transformer
        self.transformer.train()
        self.transformer.requires_grad_(False)
        print(self.transformer)

        # layers_to_remove = [4,8,12,16,20,24]
        # layers_to_remove = [4,7,10,13,16,19,22,25]
        # for idx in sorted(layers_to_remove, reverse=True):
            # del self.transformer.transformer_blocks[idx]


        lora_r = 64
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r*2,
            target_modules= ["linear_1","linear_2","linear","to_q","to_k","to_v","to_out.0",],
            lora_dropout=0.0,
            bias="none",
            # init_lora_weights='pissa',
            # use_dora=True
        )
        self.transformer = get_peft_model(self.transformer, lora_config)


        if load_ckpt:
            new_dict = torch.load('/root/sana/shrink_model_epoch_epoch=06_step_step=1000.ckpt')
            self.transformer.load_state_dict(new_dict, strict=False)

        self.pipe.vae.eval()
        for param in self.pipe.vae.parameters():
            param.requires_grad = False

        # Remove text encoder and tokenizer from the pipeline to save memory.
        del self.pipe.text_encoder
        del self.pipe.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        # Load the precomputed unconditional (null) embedding.
        null_embed = torch.load("../sana/null_embed.pt")["uncond_prompt_embeds"]
        self.register_buffer("uncond_prompt_embeds", null_embed)

    def get_timesteps(self, batch_size, device):
        num_timesteps = 1000  # Total number of timesteps
        alpha, beta_val = 1.0, 2.0
        u_beta = torch.distributions.Beta(alpha, beta_val).sample((batch_size,))
        timesteps = (u_beta * num_timesteps).to(device).long()
        return timesteps


    def training_step(self, batch, batch_idx):
        if False and self.global_step < 10000:
            self.text_prompt_ratio = 1-self.global_step/20000.0
        else:
            self.text_prompt_ratio=0.5
        self.log("text_prompt_ratio", self.text_prompt_ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            
        latents, text_encodings, _ = batch
        B = latents.size(0)

        # Determine how many samples in this batch will use text prompts
        text_prompt_samples = int(B * self.text_prompt_ratio)

        # Initialize embeddings
        uncond_text_enc = torch.zeros(B, text_encodings.size(2), text_encodings.size(3), device=latents.device)
        cond_text_enc = torch.zeros(B, text_encodings.size(2), text_encodings.size(3), device=latents.device)

        # For samples using text prompts, use the dataset's text encodings
        if text_prompt_samples > 0:
            uncond_text_enc[:text_prompt_samples] = text_encodings[:text_prompt_samples, 0]  # Unconditional embeddings
            cond_text_enc[:text_prompt_samples] = text_encodings[:text_prompt_samples, 1]    # Conditional embeddings

        # For samples not using text prompts, use the precomputed empty prompt
        empty_prompt = self.uncond_prompt_embeds[0].unsqueeze(0).repeat(B - text_prompt_samples, 1, 1)
        uncond_text_enc[text_prompt_samples:] = empty_prompt
        cond_text_enc[text_prompt_samples:] = empty_prompt

        # Scale latents and add noise.
        latents = latents * 0.41407
        # edge_latent2 = edge_latent2 * 0.41407
        noise = torch.randn_like(latents)
        timesteps = self.get_timesteps(latents.shape[0], latents.device)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)

        # Build encoder_hidden_states by concatenating unconditional and conditional embeddings.
        encoder_hidden_states = torch.cat([uncond_text_enc, cond_text_enc], dim=0)  # (2B, tokens, hidden_dim)

        # Duplicate latents and timesteps.
        model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        t_tensor = timesteps.repeat(2)
        # Process edge latent consistently.
        # edge_latent = self.transformer.patch_embed(edge_latent2)  # (B, C, H', W')
        # edge_latent = torch.cat([edge_latent, edge_latent], dim=0)   # (2B, C, H', W')

        # Forward pass through the transformer.
        noise_pred = self.transformer(
            model_input,
            encoder_hidden_states=encoder_hidden_states,
            timestep=t_tensor,
            return_dict=False
        )[0].to(torch.bfloat16)

        sigmas = self.pipe.scheduler.sigmas.to(latents.device)
        current_sigmas = sigmas[timesteps].view(latents.shape[0], 1, 1, 1)
        weighting = compute_loss_weighting(current_sigmas, scheme='none').repeat(2, 1, 1, 1)
        target = (noise - latents).repeat(2, 1, 1, 1)
        loss = torch.mean(weighting * (noise_pred - target) ** 2)


        # Log generated images every 200 steps.
        if self.global_step % 500 == 0:
            with torch.no_grad():                
                decoded_real = self.pipe.vae.decode(latents / 0.41407)
                real_image = self.pipe.image_processor.postprocess(decoded_real.sample, output_type="pil")[0]
        
                # Generate two samples: one with text prompts and one without
                latent_shape = (1, self.transformer.config.in_channels, 512 // 32, 512 // 32)
                init_noise = torch.randn(latent_shape, device="cuda", dtype=torch.bfloat16)
        
                # 1. Generate with text prompts if available
                if text_prompt_samples > 0:
                    sample_uncond = text_encodings[0, 0]  # (tokens, hidden_dim)
                    sample_cond = text_encodings[0, 1]    # (tokens, hidden_dim)
                    gen_image_text = self.generate_sample_image(
                        sample_uncond, sample_cond,
                        cfg_scale=4.5, init_latents=init_noise.clone(), 
                        use_text=True
                    )
                else:
                    gen_image_text = None
        
                # 2. Generate without text prompts
                empty_prompt = self.uncond_prompt_embeds[0]
                gen_image_no_text = self.generate_sample_image(
                    empty_prompt, empty_prompt,
                    cfg_scale=4.5, init_latents=init_noise.clone(), 
                    use_text=False
                )
        
                log_dict = {
                    "generated_no_text": wandb.Image(gen_image_no_text, caption="No Text CFG: 4.5"),
                    "real_image": wandb.Image(real_image, caption="Real Image"),
                    "train_loss": loss.item(),
                    "text_prompt_ratio": self.text_prompt_ratio
                }
                if gen_image_text is not None:
                    log_dict["generated_with_text"] = wandb.Image(gen_image_text, caption="With Text CFG: 4.5")
                
                # Let wandb handle the global step automatically.
                wandb.log(log_dict)
        
                self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                return loss

    
    def generate_sample_image(self, uncond_embeds, cond_embeds, cfg_scale=None, init_latents=None, use_text=True):
        # Save the current training mode.
        was_training = self.transformer.training
        self.transformer.eval()  # Switch to eval mode for deterministic behavior.

        with torch.no_grad():
            # Choose embeddings based on the use_text flag
            if use_text:
                uncond = uncond_embeds.unsqueeze(0)  # (1, tokens, hidden_dim)
                cond = cond_embeds.unsqueeze(0)      # (1, tokens, hidden_dim)
            else:
                # Always use the precomputed empty prompt.
                uncond = self.uncond_prompt_embeds[0].unsqueeze(0)
                cond = self.uncond_prompt_embeds[0].unsqueeze(0)
            encoder_hidden_states = torch.cat([uncond, cond], dim=0)  # (2, tokens, hidden_dim)

            latent_shape = (1, self.transformer.config.in_channels, 512 // 32, 512 // 32)
            latents = init_latents if init_latents is not None else torch.randn(latent_shape, device="cuda", dtype=torch.bfloat16)
            cfg_scale = cfg_scale if cfg_scale is not None else self.hparams.cfg_scale

            self.pipe.scheduler.set_timesteps(20)
            for t in self.pipe.scheduler.timesteps:
                model_input = torch.cat([latents, latents], dim=0)
                t_tensor = t.expand(model_input.shape[0]).to(latents.device)
                noise_pred = self.transformer(
                    model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=t_tensor,
                    return_dict=False
                )[0].to(torch.bfloat16)
                noise_uncond, noise_text = noise_pred.chunk(2)
                guided_noise = noise_uncond + cfg_scale * (noise_text - noise_uncond)
                latents = self.pipe.scheduler.step(guided_noise, t, latents, return_dict=False)[0]

            latents_scaled = latents / 0.41407
            decoded = self.pipe.vae.decode(latents_scaled)
            image = self.pipe.image_processor.postprocess(decoded.sample, output_type="pil")[0]
            self.pipe.scheduler.set_timesteps(1000)

        # Restore the original training mode.
        if was_training:
            self.transformer.train()
        torch.cuda.empty_cache()
        gc.collect()
        return image
    
    def configure_optimizers(self):
        from lion_pytorch import Lion
        return Lion(self.parameters(), lr=1e-5, weight_decay=1e-2)

