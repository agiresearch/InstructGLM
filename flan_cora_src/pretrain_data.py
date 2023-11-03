from operator import neg
from urllib.parse import urldefrag 
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
import copy


from transformers import T5TokenizerFast

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def get_loader(args, task_list, sample_numbers, split='', mode='train', 
               batch_size=16, workers=4, distributed=False):

    tokenizer = T5TokenizerFast.from_pretrained(args.backbone, max_length=args.max_text_length, do_lower_case=args.do_lower_case)

    if split == 'Cora':
        from all_graph_templates import all_tasks as task_templates
        from Cora import Cora_Dataset
        dataset = Cora_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn, drop_last=False)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,  
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        
    return loader
