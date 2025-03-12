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

# Minimal output wrapper; adjust as needed if your pipeline provides its own.
class Transformer2DModelOutput:
    def __init__(self, sample):
        self.sample = sample


def forward_ip_adapter(
    self,
    hidden_states,
    condition,
    attention_mask,
    encoder_hidden_states,
    encoder_attention_mask,
    timestep,
    height,
    width,
):
    batch_size = hidden_states.shape[0]

    # 1. Modulation
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
    ).chunk(6, dim=1)

    # 2. Self Attention
    norm_hidden_states = self.norm1(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

    attn_output = self.attn1(norm_hidden_states)
    hidden_states = hidden_states + gate_msa * attn_output

    # 3. Cross Attention
    if self.attn2 is not None:
        attn_output = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states
        attn_output = self.attn3(
            hidden_states,
            encoder_hidden_states=condition,
            # attention_mask=encoder_attention_mask2,
        )
        attn_output = self.zero_linear(attn_output)
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
    ff_output = self.ff(norm_hidden_states)
    ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
    hidden_states = hidden_states + gate_mlp * ff_output

    return hidden_states


def forward_c(
    self,
    hidden_states: torch.Tensor,
    condition: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 1. Input
    batch_size, num_channels, height, width = hidden_states.shape
    p = self.config.patch_size
    post_patch_height, post_patch_width = height // p, width // p

    hidden_states = self.patch_embed(hidden_states)

    timestep, embedded_timestep = self.time_embed(
        timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
    )

    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    encoder_hidden_states = self.caption_norm(encoder_hidden_states)

    # 2. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        def create_custom_forward(module, return_dict=None):
            def custom_forward(*inputs):
                if return_dict is not None:
                    return module(*inputs, return_dict=return_dict)
                else:
                    return module(*inputs)
            return custom_forward

        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
        for block in self.transformer_blocks:
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                condition,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_height,
                post_patch_width,
                **ckpt_kwargs,
            )
    else:
        counter = 0
        for block in self.transformer_blocks:
            if counter < 14:
                hidden_states = block(
                    hidden_states,
                    condition,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    # condition is omitted for these blocks
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
            counter += 1

    # 3. Normalization
    shift, scale = (
        self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
    ).chunk(2, dim=1)
    hidden_states = self.norm_out(hidden_states)

    # 4. Modulation
    hidden_states = hidden_states * (1 + scale) + shift
    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify
    hidden_states = hidden_states.reshape(
        batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
    )
    hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
    output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

class TwoLayerSwiGLUMLP(nn.Module):
    def __init__(self, input_dim=1152, output_dim=1152, hidden_dim_factor=2):
        super().__init__()
        hidden_dim = input_dim * hidden_dim_factor  # hidden_dim = 2304
        # Combined linear layer: outputs both projections in one go.
        self.fc = nn.Linear(input_dim, 2 * hidden_dim)  # output shape: (batch, 4608)
        # Final projection expects input of half the hidden dimension (i.e., 1152)
        self.w3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # Compute both projections with one linear layer
        x_proj = self.fc(x)  # shape: (batch, 4608)
        # Split the projections into two halves
        x1, x2 = x_proj.chunk(2, dim=-1)  # each of shape: (batch, 2304)
        # Apply activation to the second half
        x2 = F.silu(x2)
        # Elementwise multiplication: shape remains (batch, 2304)
        hidden = x1 * x2
        # Reduce the dimension by half (choose the first half)
        hidden = hidden.chunk(2, dim=-1)[0]  # shape: (batch, 1152)
        # Final projection
        out = self.w3(hidden)
        return out

def compute_loss_weighting(sigmas, scheme='reciprocal'):
    """
    Computes a loss weighting tensor based on the noise levels (sigmas).

    Args:
        sigmas (torch.Tensor): Tensor of noise levels with shape [batch_size, 1, 1, 1].
        scheme (str): Weighting scheme. Options:
                      - 'none': No weighting (returns ones).
                      - 'reciprocal': Returns 1/(sigma^2) (with a small epsilon for stability).
                      - 'sigma_sqrt': Returns sqrt(sigma).

    Returns:
        torch.Tensor: Weighting tensor of the same shape as sigmas.
    """
    if scheme == 'none':
        return torch.ones_like(sigmas)
    elif scheme == 'reciprocal':
        return 1.0 / (sigmas ** 2 + 1e-8)  # Avoid division by zero
    elif scheme == 'sigma_sqrt':
        return torch.sqrt(sigmas)
    else:
        return torch.ones_like(sigmas)

class ControlNetBlock(nn.Module):
    def __init__(self, target_block, channels, first_block=False):
        super().__init__()
        self.og_block = target_block
        self.first_block = first_block
        self.trainable_block = copy.deepcopy(target_block)
        for param in self.trainable_block.parameters():
            param.requires_grad = True
        self.zero_linear1 = nn.Linear(channels, channels).to(next(target_block.parameters()).device)
        nn.init.zeros_(self.zero_linear1.weight)
        nn.init.zeros_(self.zero_linear1.bias)
        self.zero_linear2 = nn.Linear(channels, channels).to(next(target_block.parameters()).device)
        nn.init.zeros_(self.zero_linear2.weight)
        nn.init.zeros_(self.zero_linear2.bias)

    def forward(self, x, c, *args, **kwargs):
        c = self.zero_linear1(c)
        c = x + c
        c_next = self.trainable_block(c, *args, **kwargs)
        c = self.zero_linear2(c_next)
        x = self.og_block(x, *args, **kwargs)
        return x + c, c_next

class ControlNetFineTuner(pl.LightningModule):
    def __init__(self, load_ckpt, cfg_scale=3.0, text_prompt_ratio=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.text_prompt_ratio = text_prompt_ratio

        self.c_mlp = TwoLayerSwiGLUMLP(input_dim=2176, output_dim=1152*4, hidden_dim_factor=4)

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

        # Replace forward with our custom forward.
        self.transformer.forward = forward_c.__get__(self.transformer)
        for i in range(len(self.transformer.transformer_blocks)):
            if i < 14:
                self.transformer.transformer_blocks[i].forward = forward_ip_adapter.__get__(self.transformer.transformer_blocks[i])
                self.transformer.transformer_blocks[i].attn3 = copy.deepcopy(self.transformer.transformer_blocks[i].attn2)
                for param in self.transformer.transformer_blocks[i].attn3.parameters():
                    param.requires_grad = True
                self.transformer.transformer_blocks[i].zero_linear = nn.Linear(1152, 1152).to(next(self.transformer.transformer_blocks[i].parameters()).device)
                nn.init.zeros_(self.transformer.transformer_blocks[i].zero_linear.weight)
                nn.init.zeros_(self.transformer.transformer_blocks[i].zero_linear.bias)
                # self.transformer.transformer_blocks[i].load_state_dict(old_dict,strict=True)
                

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
        if self.global_step < 10000:
            self.text_prompt_ratio = 1-self.global_step/20000.0
        else:
            self.text_prompt_ratio=0.5
        self.log("text_prompt_ratio", self.text_prompt_ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            
        latents, text_encodings, image_encoding = batch
        image_encoding = self.c_mlp(image_encoding)
        image_encoding = image_encoding.reshape(image_encoding.shape[0],4,1152)
        # print(image_encoding.shape) [6, 1, 1152]
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
        image_encoding = torch.cat([image_encoding, image_encoding], dim=0)

        # Duplicate latents and timesteps.
        model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        t_tensor = timesteps.repeat(2)
        # Process edge latent consistently.
        # edge_latent = self.transformer.patch_embed(edge_latent2)  # (B, C, H', W')
        # edge_latent = torch.cat([edge_latent, edge_latent], dim=0)   # (2B, C, H', W')

        # Forward pass through the transformer.
        noise_pred = self.transformer(
            model_input,
            image_encoding,
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
                image_encoding = image_encoding[0].unsqueeze(0)
                
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
                        sample_uncond, sample_cond, image_encoding,
                        cfg_scale=4.5, init_latents=init_noise.clone(), 
                        use_text=True
                    )
                else:
                    gen_image_text = None
        
                # 2. Generate without text prompts
                empty_prompt = self.uncond_prompt_embeds[0]
                gen_image_no_text = self.generate_sample_image(
                    empty_prompt, empty_prompt, image_encoding, 
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

    
    def generate_sample_image(self, uncond_embeds, cond_embeds, image_encoding, cfg_scale=None, init_latents=None, use_text=True):
        image_encoding = torch.cat([image_encoding, image_encoding], dim=0)  # (2, tokens, hidden_dim)

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
                    image_encoding,
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
