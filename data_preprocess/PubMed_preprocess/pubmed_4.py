from collections import defaultdict  #
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

dataset = Planetoid('./pubmed_data', 'PubMed',split='public')
data = dataset[0]

# Only for small-pubmed, i.e. 20 training nodes for each category. We use the widely-used fixed split of official pubmed.
#classification = list(range(0, 60))
#validation= list(range(60,560))
#transductive = list(range(18717, 19717))
#save_pickle(transductive, 'small_final_pub_transductive.pkl')
#save_pickle(validation, 'small_final_pub_valid.pkl')
#save_pickle(classification, 'small_final_pub_classification.pkl')




### real feature
bbb = torch.zeros(32100, 500)
real_feature = torch.cat([bbb, data.x], dim=0)
print(real_feature.dtype)

import copy
xx=copy.deepcopy(data.x)
from sklearn.preprocessing import normalize
xx=normalize(xx, norm="l1")
print(torch.tensor(xx))
norm_real_feature=torch.cat([bbb,torch.tensor(xx)], dim=0)
print(norm_real_feature.dtype)

save_pickle(real_feature, 'final_pub_real_feature.pkl')
save_pickle(norm_real_feature, 'final_norm_pub_real_feature.pkl')

check = {0: 'experimental', 1: 'second', 2: 'first'}
label_map = {}
for i in range(19717):
    label_map[i] = check[data.y[i].item()]
save_pickle(label_map, 'final_pub_label_map.pkl')


re_id = {}
for i in range(19717):
    re_id[i] = 32100+i
save_pickle(re_id, 'final_pub_re_id.pkl')

all = []
for i in range(len(data.edge_index.T)):
    e = list(np.array(data.edge_index.T[i]))
    if e in all or e[::-1] in all:
        pass
    else:
        all.append(e)
print(len(all))  # 44324 = 88648/2

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


save_pickle(L1, 'final_pub_L1.pkl')