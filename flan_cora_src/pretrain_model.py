import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_p5 import P5

class P5Pretraining(P5):
    def __init__(self, config):
        super().__init__(config)

        self.losses = self.config.losses.split(',')

    def train_step(self, batch):    #batch is a dict

        device = next(self.parameters()).device

        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)
        lm_labels = batch["target_ids"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        output = self(  #这里本质也是调用forward
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask   #注意一下loss的size

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)   #这里之后变1维的了

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        results = {}     #Real output for our model

        results['loss'] = (loss * loss_weights).mean()    #这个loss_weight是per_batch(per_example)的
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        return results

    @torch.no_grad()
    def valid_step(self, batch):   #真正的inference是不会调用本函数的，必须generate
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)

        lm_labels = batch["target_ids"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        output = self(           #没有whole_word_embed的时候怎么做输出的？   ；  本质上就只是过了一次forward
            input_ids=input_ids,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        if 'rating' in self.losses:
            output = self.generate(
                input_ids=input_ids  #shape[0] is Batch_Size
            )

            generated_score = self.tokenizer.batch_decode(output, skip_special_tokens=True)

            results['rating_pred'] = generated_score

        return results

    @torch.no_grad()
    def generate_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)

        output = self.generate(
            input_ids=input_ids,
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return generated_sents

    @torch.no_grad()
    def g_step(self, batch,real=None):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
       # whole_word_ids = batch['whole_word_ids'].to(device)
#        if real is not None:
        real=real.to(device)


        output = self.generate(
            input_ids=input_ids,
            real_feature=real,
           # temperature=0.1,
            max_new_tokens=9
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)   #所以decode出来是不带有结束符号的

        return generated_sents
