from transformers import T5TokenizerFast as T
import csv
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

'''
Set seeds
'''
seed = 2022
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name='ogbn-arxiv')

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
g = dataset[0]
classification=list(np.array(train_idx))
validation=list(np.array(valid_idx))
transductive=list(np.array(test_idx))

tt=T.from_pretrained('google/flan-t5-large')

paperID_id={}


csv_reader = csv.reader(open("nodeidx2paperid.csv"))
next(csv_reader)
for line in csv_reader:
    if line[1] not in paperID_id:
        paperID_id[line[1]]=int(line[0])

num=0
abs=[]
tit=[]
n=0

text_feature={}
with open('titleabs.tsv','r',encoding='utf-8') as f:
    next(f)  # pass the first line
    for line in f:
        n+=1
        print(n)
        if n==179719:
            continue
        line = line.strip('\n').split('\t')
        if line[0] in paperID_id.keys():  
            ll = len(tt.encode(line[2]))
            oo = len(tt.encode(line[1]))
            text_feature[paperID_id[line[0]]]=[line[1]]   # title
            if ll>450:  # Max token limit of Flan-t5 is 512
                num+=1
                text_feature[paperID_id[line[0]]].append(tt.decode(tt.encode(line[2])[:449]))  # abstract
            else:
                text_feature[paperID_id[line[0]]].append(line[2])   # abstract
            abs.append(ll)
            tit.append(oo)
print()
print(len(text_feature)) #169343

print()
print(len(paperID_id))  #169343
print()
print(num) 
print()

print(max(abs),max(tit)) 
print(min(abs),min(tit)) 
print(np.mean(abs),np.mean(tit)) 
print(np.median(abs),np.median(tit)) 

check={}   # setup label map
n=0
csv_reader = csv.reader(open("labelidx2arxivcategeory.csv"))
next(csv_reader)
for line in csv_reader:
    check[int(line[0])]=line[2]

print(len(check))
print(check)
print()

label_map={}
for i in range(169343):
    label_map[i]=check[g.y[i].item()]
print(g.y[0].item(),label_map[0])
print()

re_id={}
for i in range(169343):
    re_id[i]=32100+i

#OGB feature (dim==128)
#Same for GIANT (dim==768) feature once the giant.pkl is downloaded.
bbb=torch.zeros(32100,128)
real_feature=torch.cat([bbb,g.x],dim=0)


all=[]   #To get all edges
temp=g.edge_index.T
for i in tqdm(range(len(temp))):
    e=list(np.array(temp[i]))

    all.append(e)

L1={}
for thin in tqdm(range(len(all))):  
    thing=all[thin]
    if thing[0] in L1:
        L1[thing[0]].append(thing[1])
    else:
        L1[thing[0]]=[thing[1]]
    if thing[1] in L1:
        L1[thing[1]].append(thing[0])
    else:
        L1[thing[1]]=[thing[0]]

print('L1 finished')

save_pickle(L1, 'L1.pkl') # 1-hop neighbors infomation
save_pickle(text_feature, 'node_feature.pkl')
save_pickle(label_map, 'label_map.pkl')
save_pickle(re_id, 're_id.pkl')
save_pickle(real_feature, 'real_feature.pkl') # OGB
##save_pickle(giant_feature, 'L_giant.pkl') # GIANT
save_pickle(transductive, 'transductive.pkl') #TEST
save_pickle(validation, 'validation.pkl') #VAL
save_pickle(classification, 'classification.pkl') #TRAIN