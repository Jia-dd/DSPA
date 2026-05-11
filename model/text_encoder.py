import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, pretrained_clip):
        super().__init__()
        self.positional_embedding = pretrained_clip.positional_embedding
        self.transformer = pretrained_clip.transformer
        self.ln_final = pretrained_clip.ln_final
        self.text_projection = pretrained_clip.text_projection
       
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(prompts.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(prompts.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
