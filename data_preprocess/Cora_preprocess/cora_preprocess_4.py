import copy
from collections import defaultdict
import os
import torch
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
from tqdm import tqdm


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


from torch_geometric.datasets import Planetoid

'''
Set seeds
'''
seed = 2022
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = Planetoid('./cora_data', 'Cora',split='public')
data = dataset[0]

bbb = torch.zeros(32100, 1433)
real_feature = torch.cat([bbb, data.x], dim=0)
save_pickle(real_feature, 'final_cora_real_feature.pkl')


check = {0: 'theory', 1: 'reinforcement learning', 2: 'genetic algorithms', 3: 'neural networks',
         4: 'probabilistic methods', 5: 'case based', 6: 'rule learning'}
label_map = {}
for i in range(2708):
    label_map[i] = check[data.y[i].item()]
save_pickle(label_map, 'final_cora_label_map.pkl')

re_id = {}
for i in range(2708):
    re_id[i] = 32100+i
save_pickle(re_id, 'final_cora_re_id.pkl')

all = []
for i in range(len(data.edge_index.T)):
    e = list(np.array(data.edge_index.T[i]))
    if e in all or e[::-1] in all:
        pass
    else:
        all.append(e)
print(len(all))

L1={}
for thing in all:
    if thing[0] in L1:
        L1[thing[0]].append(thing[1])
    else:
        L1[thing[0]]=[thing[1]]
    if thing[1] in L1:
        L1[thing[1]].append(thing[0])
    else:
        L1[thing[1]]=[thing[0]]

print('L1 finished')

save_pickle(L1, 'final_cora_L1.pkl')  # Generated according to official Cora Dataset
