# models/finetuner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import gc
import pytorch_lightning as pl
from diffusers import SanaPipeline, DPMSolverMultistepScheduler
from typing import Optional, Dict, Any

# Minimal output wrapper; adjust as needed if your pipeline provides its own.
class Transformer2DModelOutput:
    def __init__(self, sample):
        self.sample = sample

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
                hidden_states, condition = block(
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

class CBlock(nn.Module):
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
                self.transformer.transformer_blocks[i] = CBlock(
                    self.transformer.transformer_blocks[i], 1152, first_block=(i == 0)
                )

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
        null_embed = torch.load("null_embed.pt")["uncond_prompt_embeds"]
        self.register_buffer("uncond_prompt_embeds", null_embed)

    def get_timesteps(self, batch_size, device):
        num_timesteps = 1000  # Total number of timesteps
        alpha, beta_val = 1.0, 2.0
        u_beta = torch.distributions.Beta(alpha, beta_val).sample((batch_size,))
        timesteps = (u_beta * num_timesteps).to(device).long()
        return timesteps

    def training_step(self, batch, batch_idx):
        # Insert your training logic here.
        # This is a placeholder returning a dummy loss.
        loss = torch.tensor(0.0, device=self.device)
        return loss

    def configure_optimizers(self):
        from lion_pytorch import Lion
        return Lion(self.parameters(), lr=1e-5, weight_decay=1e-2)
