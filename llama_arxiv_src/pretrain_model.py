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
    def g_step(self, in_embeds, attention_mask):   # For Inference text Generation
        # Notably, our input here is numberical inputs_embeds, i.e. we already map inputs_ids to embeddings in pretrain.py via 'first_model'
        self.eval()
        device = next(self.parameters()).device
        in_embeds=in_embeds.to(device)
        attention_mask=attention_mask.to(device)

        output = self.generate(    
             # When the text input format is inputs_embeds rather than inputs_ids, the generate function will automatically help generate the first token as BOS token during inference.
             # Thus ensures consistency in the pipeline between training and inference.
            inputs_embeds=in_embeds,
            attention_mask=attention_mask,
            max_new_tokens=9
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)   

        return generated_sents
