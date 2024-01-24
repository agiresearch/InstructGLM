import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
import pandas as pd

import pickle
import copy

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

a=load_pickle('cora_map.pkl')
aa=copy.deepcopy(a)
rest=load_pickle('cora_rest.pkl')
print(rest)
print(len(rest))  # 27 = 7*2 + 3*3 + 1*4

# Since the number of repeated nodes is very small,
# we just simply obtain the following 'repeat' list by manual check via observing the 'rest' list
# for an example, there are 2 items [137, 2, [2150,2183]] and [2144, 2, [2150, 2183]],
# which means the nodes indexed by 137 and 2144 in the official Cora dataset are in the same repeated group.
repeat=[[137,2144],[283,1260],[366,1127,1995],[503,574,2348],[995,1338,2102,2103],[709,1897],[950,1495],[1020,2076],
        [1040,1719,1720],[1033,1586],[1031,2205]]
for rr in repeat:
    for ii in range(len(rr)):
        aa[rr[ii]]=[a[rr[ii]][ii]]   # Alignment

for k,v in aa.items():
    aa[k]=v[0]

bb={}
for k,v in aa.items():
    bb[v]=k

save_pickle(aa,'cora_tape.pkl')
save_pickle(bb,'tape_cora.pkl')

cora_tape=load_pickle('cora_tape.pkl')
print(cora_tape)
tape_cora=load_pickle('tape_cora.pkl')    # we need this file for next step preprocessing.
print(tape_cora)

title=load_pickle('cora_title.pkl')
print(title[0])
abss=load_pickle('cora_abs.pkl')
print(abss[0])

final_node_feature={}
lab,lti=[],[]
from transformers import T5TokenizerFast as T
tt=T.from_pretrained('google/flan-t5-base')

from tqdm import tqdm
for i in tqdm(range(2708)):
    tttt=title[cora_tape[i]]
    aaaa=abss[cora_tape[i]]
    l_t=len(tt.encode(tttt))
    if l_t>100:
        final_node_feature[i]=[tt.decode(tt.encode(tttt)[:90],skip_special_tokens=True)]
    else:
        final_node_feature[i]=[tttt]

    l_a=len(tt.encode(aaaa))
    if l_a > 450:
        temp = tt.decode(tt.encode(aaaa), skip_special_tokens=True)
        while len(tt.encode(temp)) > 466:
            temp = tt.decode(tt.encode(temp)[:465], skip_special_tokens=True)
        final_node_feature[i].append(temp)
    else:
        final_node_feature[i].append(aaaa)
    lab.append(l_a)
    lti.append(l_t)

save_pickle(final_node_feature,'final_cora_node_feature.pkl')

