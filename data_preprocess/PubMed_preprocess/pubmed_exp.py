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

a=load_pickle('pubmed_map.pkl')
repeat=[[6718,6719],[10009,14427],[11332,15329,18951]]


import copy
aa=copy.deepcopy(a)
for rr in repeat:
    for ii in range(len(rr)):
        aa[rr[ii]]=[a[rr[ii]][ii]]

for k,v in aa.items():
    if len(v)>1:
        print('nooooooooooooo')
print('nothing')

for k,v in aa.items():
    aa[k]=v[0]

bb={}
for k,v in aa.items():
    bb[v]=k

#save mapping
#save_pickle(aa,'pubmed_tape.pkl')
#save_pickle(bb,'tape_pubmed.pkl')
#




pubmed_tape=load_pickle('pubmed_tape.pkl')
print(len(pubmed_tape))
print(pubmed_tape[1],pubmed_tape[2],pubmed_tape[3])
print('seeeeeeeee')
tape_pubmed=load_pickle('tape_pubmed.pkl')
print(len(tape_pubmed))




title=load_pickle('pubmed_title.pkl')
print(title[0])
abss=load_pickle('pubmed_abs.pkl')
print(abss[0])
print(len(title),len(abss))

final_node_feature={}
lab,lti=[],[]
from transformers import T5TokenizerFast as T
tt=T.from_pretrained('google/flan-t5-base')
from tqdm import tqdm
for i in tqdm(range(19717)):
    tttt=title[pubmed_tape[i]]
    aaaa=abss[pubmed_tape[i]]
    l_t=len(tt.encode(tttt))

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

print(len(final_node_feature))
print(max(lab),max(lti)) #1270,122
print(min(lab),min(lti)) #1,1
print(np.mean(lab),np.mean(lti)) #385,28
print(np.median(lab),np.median(lti)) #384,27

print(torch.topk(torch.tensor(lti),10))
print(torch.topk(torch.tensor(lab),100))

save_pickle(final_node_feature,'full_final_pubmed_node_feature.pkl')
print(len(final_node_feature))