import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_p5 import P5

class P5Pretraining(P5):
    def __init__(self, config):
        super().__init__(config)

        self.losses = self.config.losses.split(',')

    @torch.no_grad()
    def g_step(self, in_embeds,attention_mask):#batch, real):        #forward pipeline得重写，现在这里好乱
        self.eval()
        device = next(self.parameters()).device
        #input_ids = batch['input_ids'].to(device)
        #attention_mask=batch['attn_mask'].to(device)
        #real=real.to(device)
        in_embeds=in_embeds.to(device)
        attention_mask=attention_mask.to(device)

####33#######33###########这下面不能有问题了
        output = self.generate(
            inputs_embeds=in_embeds,
            attention_mask=attention_mask,
            max_new_tokens=9
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)   #所以decode出来是不带有结束符号的

        return generated_sents
