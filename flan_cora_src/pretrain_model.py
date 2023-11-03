import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_flan import GLM

class InstructGLM(GLM):
    def __init__(self, config):
        super().__init__(config)

        self.losses = self.config.losses.split(',')


    @torch.no_grad()
    def g_step(self, batch,real=None):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        real=real.to(device)


        output = self.generate(
            input_ids=input_ids,
            real_feature=real,
            max_new_tokens=9
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)   

        return generated_sents
