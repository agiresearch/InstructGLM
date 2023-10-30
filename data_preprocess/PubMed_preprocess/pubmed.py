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

# return pubmed dataset as pytorch geometric Data object together with 60/20/20 split, and list of pubmed IDs


def get_pubmed_casestudy(corrected=False, SEED=0):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    #data_X = normalize(data_X, norm="l1")其实要做

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('pubmed_dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    data.x = torch.tensor(data_X)
    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    if corrected:
        is_mistake = np.loadtxt(
            'pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        data.train_id = [i for i in data.train_id if not is_mistake[i]]
        data.val_id = [i for i in data.val_id if not is_mistake[i]]
        data.test_id = [i for i in data.test_id if not is_mistake[i]]

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_pubid


def parse_pubmed():
    path = './PubMed_orig/data/'

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(path + 'Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - \
                1  # subtract 1 to zero-count
            data_Y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    with open(path + 'Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_A, data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_pubmed(use_text=True, seed=0):
    data, data_pubid = get_pubmed_casestudy(SEED=seed)
   # print(data_pubid)
    if not use_text:
        return data, None

    f = open('./PubMed_orig/pubmed.json')
    pubmed = json.load(f)
    #print(len(pubmed))
    df_pubmed = pd.DataFrame.from_dict(pubmed)
    L=list(df_pubmed['PMID'])
    wrong=[]
    for gg in range(19717):
        if L[gg]!=data_pubid[gg]:
            wrong.append(gg)
    print(len(wrong))
    print(L[2459])
    print(data_pubid[2459])
    print(df_pubmed[2459:2450])
    print(wrong)
    #print(list(df_pubmed['PMID'])==data_pubid)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    title=[]
    abss=[]
    for ti, ab in zip(TI, AB):
       # t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        #text.append(t)
        title.append(ti)
        abss.append(ab)
    return data, title, abss, data_pubid

data, title, abss, data_pubid = get_raw_text_pubmed()
#3728 13844 16247 ;;;;1,0,2 ;;;;;  1:type2 0:experimental 2:type1
print(data_pubid[3728],data_pubid[13844],data_pubid[16247])



tape_pubmed=load_pickle('tape_pubmed.pkl')


p_classification=list(data.train_id)
p_transductive=list(data.test_id)
classification,transductive=[],[]
for i in p_classification:
    classification.append(tape_pubmed[i])
for j in p_transductive:
    transductive.append(tape_pubmed[j])
#save_pickle(transductive, 'big_final_pub_transductive.pkl')
#save_pickle(classification, 'big_final_pub_classification.pkl')
print(len(classification))#11830
print(len(transductive))#3944

print(data)
print(data.x)
print(len(title),len(abss))
#save_pickle(abss,'pubmed_abs.pkl')
#save_pickle(title,'pubmed_title.pkl')

from torch_geometric.datasets import Planetoid
dataset = Planetoid('./pubmed_data', 'PubMed',split='public')
g=dataset[0]
print(len(set(g.x.sum(1).tolist())))   #19660
print(len(set(data.x.sum(1).tolist())))
print(title[19716])
import re
print(abss[19716])

#same=0
#a={}
#from tqdm import tqdm
#for j in tqdm(range(19717)):
 #   print(j)
  #  tt=torch.tensor(list(set(g.x[j].tolist())))
   # a[j]=[]
    #for i in range(19717):
     #   if tt.equal(torch.tensor(list(set(data.x[i].tolist())))):
      #      a[j].append(i)
       #     same+=1
print()
#print(same) #19727
a=load_pickle('pubmed_map.pkl')
rest=load_pickle('pubmed_rest.pkl')
print(len(a))  #19717

for k,v in a.items():
    if len(v)>1:
        print((k,len(v),v))
       # rest.append(k)
print(rest)
#save_pickle(a,'pubmed_map.pkl')
#save_pickle(rest,'pubmed_rest.pkl')

