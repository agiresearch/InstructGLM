import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_llama import GLM

class InstructGLM(GLM):
    def __init__(self, config):
        super().__init__(config)

        self.losses = self.config.losses.split(',')

    @torch.no_grad()
    def g_step(self, in_embeds, attention_mask):
        self.eval()
        device = next(self.parameters()).device
        in_embeds=in_embeds.to(device)
        attention_mask=attention_mask.to(device)

        output = self.generate(
            inputs_embeds=in_embeds,
            attention_mask=attention_mask,
            max_new_tokens=9
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)   

        return generated_sents
