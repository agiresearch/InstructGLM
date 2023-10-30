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

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

a=load_pickle('cora_map.pkl')
repeat=[[137,2144],[283,1260],[366,1127,1995],[503,574,2348],[995,1338,2102,2103],[709,1897],[950,1495],[1020,2076],
        [1040,1719,1720],[1033,1586],[1031,2205]]


rest=load_pickle('cora_rest.pkl')

all=load_pickle('cora_all.pkl')
simple=load_pickle('cora_simple.pkl')

import copy
aa=copy.deepcopy(a)
for rr in repeat:
    for ii in range(len(rr)):
        aa[rr[ii]]=[a[rr[ii]][ii]]

for k,v in aa.items():
    aa[k]=v[0]

bb={}
for k,v in aa.items():
    bb[v]=k

cora_tape=load_pickle('cora_tape.pkl')
print(cora_tape)
tape_cora=load_pickle('tape_cora.pkl')
print(tape_cora)

title=load_pickle('cora_title.pkl')
print(title[0])
abss=load_pickle('cora_abs.pkl')
print(abss[0])
print(len(title),len(abss))

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
    if l_a < -450:
        temp = tt.decode(tt.encode(aaaa), skip_special_tokens=True)
        while len(tt.encode(temp)) > 466:
            temp = tt.decode(tt.encode(temp)[:465], skip_special_tokens=True)
        final_node_feature[i].append(temp)
    else:
        final_node_feature[i].append(aaaa)
    lab.append(l_a)
    lti.append(l_t)


#save_pickle(final_node_feature,'full_final_cora_feature.pkl')
print(len(final_node_feature))
print(max(lab),max(lti)) #1293,302
print(min(lab),min(lti)) #1,1
print(np.mean(lab),np.mean(lti)) #169,17
print(np.median(lab),np.median(lti)) #162, 17

print(torch.topk(torch.tensor(lti),10))
print(torch.topk(torch.tensor(lab),100))

##save_pickle(final_node_feature,'full_final_cora_node_feature.pkl')
#print(final_node_feature)






