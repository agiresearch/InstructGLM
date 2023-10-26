import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint 

from utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pformat

proj_dir = Path(__file__).resolve().parent.parent

_use_native_amp = False
_use_apex = False


_use_native_amp = True
from torch.cuda.amp import autocast


class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.verbose = True     # Only the main GPU's verbose == True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

    def create_config(self):
        from transformers import T5Config

        config_class = T5Config
 

        config = config_class.from_pretrained(self.args.backbone)

        args = self.args

        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout

        config.losses = args.losses

        return config

    def create_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU {self.args.gpu}')

        model_name = self.args.backbone

        model = model_class.from_pretrained(
            model_name,
            config=config,
            **kwargs
        )
        return model

    def create_tokenizer(self, **kwargs):
        from transformers import T5TokenizerFast

        tokenizer_class = T5TokenizerFast

        tokenizer_name = self.args.backbone
        if True:
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case,
                **kwargs
                )

        return tokenizer

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        if 'adamw' in self.args.optim:
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup

            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)
            
            if self.verbose:
                print("Batch per epoch: %d" % batch_per_epoch)
                print("Total Iters: %d" % t_total)
                print('Warmup ratio:', warmup_ratio)
                print("Warm up Iters: %d" % warmup_iters)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
#            lr_scheduler = get_linear_schedule_with_warmup(   # We also provide scheduler here despite we didn't use such a linear scheduler in our implementations.
 #               optim, warmup_iters, t_total)   

        else:
            optim = self.args.optimizer(
                list(self.model.parameters()), self.args.lr)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')
        results = self.model.load_state_dict(state_dict, strict=True)  
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)
        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)
