from urllib.parse import urldefrag ##
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
from copy import deepcopy

from transformers import T5Tokenizer, T5TokenizerFast
#from tokenization import P5Tokenizer, P5TokenizerFast

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

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

split='beauty'
item_l=12101
user_l=22363

sequential_data = ReadLineFromFile(os.path.join('data', split, 'sequential_data.txt'))  
item_count = defaultdict(int)   # default值为0
user_items = defaultdict()
        

#        mmmm=-1
for line in sequential_data:    #这里都是id数字化了, 但是是数字字符串
    user, items = line.strip().split(' ', 1)
    items = items.split(' ')
    items = [int(item) for item in items]
    user=int(user)
    user_items[user] = items                 # here is what we need.   这里的user还是字符

    for item in items:
        item_count[item] += 1 

all_item = list(item_count.keys())
all_user=list(user_items.keys())
#print(all_item)
#print(all_user)

LL1,LL2=[],[]
L1={}  
LL1.append([])#不索用0
L2={}
LL2.append([])


#先建立L1
#key: node_id, value: list of node_id
for i in range(item_l+1,(item_l+user_l+1)):
    L1[i]=user_items[i-item_l]
for i in range(1,item_l+1):
    L1[i]=[]
for uu,ii in user_items.items():
    for i in ii:
        L1[i].append(uu+item_l)
#L1建立完毕
print('L1 finished')

#建立L2
for i in range(1,(item_l+user_l+1)):
    L2[i]=[]
    for e in L1[i]:
        for t in L1[e]:
            if t!=i:
                L2[i].append([e,t])
print('L2 finished')

#建立L3
#for i in tqdm(range(1,(item_l+user_l+1))):   #这个不建立了，按照此pipeline，需要的时候直接当场建
 #   L3[i]=[]
  #  for e in L2[i]:
   #     for t in L1[e[1]]:
    #        if t!=i and t!=e[0]:
     #           L3[i].append(e+[t])
#print('L3 finished')

for i in tqdm(range(1,(item_l+user_l+1))):
    LL1.append(L1[i])
    LL2.append(L2[i])
 #   LL3.append(L3[i])




#save_pickle(LL3, 'L3.pkl')
save_pickle(LL1, 'L1.pkl')
save_pickle(LL2, 'L2.pkl')


print('Finished saving 2 Lists')






    
    





