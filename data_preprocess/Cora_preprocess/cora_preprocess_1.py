import numpy as np
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs


def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('./cora_dataset', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid


def parse_cora():
    path = './Cora/cora'
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path+'.cites/cora'), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text=True, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open('./Cora/mccallum/cora/papers')as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        fn=fn.split(':',1)
        if len(fn)==2:
            pid_filename[pid] = fn[0]+'_'+fn[1]
        else:
            pid_filename[pid]=fn[0]


    path = './Cora/mccallum/cora/extractions/'
    text = []
    abss=[]
    title=[]
    print(len(data_citeid))
    iii=0
    print(data_citeid[2150],data_citeid[2183])

    for pid in data_citeid:
        iii+=1
        fn = pid_filename[pid]    #64
        if fn in ['http_##www.cs.orst.edu:80#~tadepall#research#papers#speedup-learning.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr279.ps',
                  'http_##c.gp.cs.cmu.edu:5103#afs#cs.cmu.edu#project#reinforcement#papers#deng.multires.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr255.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr313.ps.gz',
                  'http_##www.stat.washington.edu:80#tech.reports#tr329.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr285.ps',
                  'http_##www.cs.orst.edu:80#~tadepall#research#papers#auto-exploratory.ps',
                  'http_##www.aic.nrl.navy.mil:80#~aha#aaai95-fss#papers#boerner.ps.Z',
                  'http_##pegasus.ece.utexas.edu:80#~ismail#ruleordering_IEEE_ICNN97.ps.Z',
                  'http_##www.stat.washington.edu:80#tech.reports#bic.ps',
                  'http_##www-cad.eecs.berkeley.edu:80#HomePages#aml#publications#ml92_def.ps.gz',
                  'http_##merlin.mbcr.bcm.tmc.edu:8001#bcdusa#Curric#MulAli#mulali.ps.gz',
                  'http_##www.cs.orst.edu:80#~tadepall#research#papers#h-ebrl.ps',
                  'http_##www.cs.orst.edu:80#~tadepall#research#papers#h-learning-tr.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr317.ps',
                  'http_##www.cs.rochester.edu:80#u#mccallum#mccallum-nips96.ps.gz',
                  'http_##www.stat.washington.edu:80#tech.reports#tr295.ps',
                  'http_##www.cs.orst.edu:80#~tadepall#research#papers#local-linear-regression.ps',
                  'http_##or.eng.tau.ac.il:7777#ml4.ps.Z',
                  'http_##enuxsa.eas.asu.edu:80#~ihrig#t.ps.Z',
                  'http_##www.stat.washington.edu:80#tech.reports#tr281.ps',
                  'http_##e9.ius.cs.cmu.edu:8000#~parag#research#papers#alvinn_tech_rep#alvinn_tech_rep.ps.gz',
                  'ftp_##ftp.cs.helsinki.fi#pub#Reports#by_Project#Cosco#NEULA:_A_hybrid_neural-symbolic_expert_system_shell.ps.gz',
                  'http_##www.stat.washington.edu:80#tech.reports#tr268.ps',
                  'http_##pi1093.kub.nl:2080#~ilk#papers#ilk9703.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr292.ps',
                  'http_##www.cs.orst.edu:80#~tadepall#research#papers#determinations.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr310.ps',
                  'http_##www.aic.nrl.navy.mil:80#papers#1997#AIC-97-004.ps.Z',
                  'http_##www.stat.washington.edu:80#tech.reports#tr274.ps',
                  'http_##www.cs.gmu.edu:80#research#gag#papers#sarma-ppsn96.ps',
                  'http_##www.cs.orst.edu:80#~tadepall#research#papers#unsupervised-speedup-learning.ps',
                  'ftp_##cns-ftp.bu.edu#pub#diana#GroBra:95.ps.gz',
                  'http_##www.stat.washington.edu:80#tech.reports#tr308.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr262.ps',
                  'http_##www.cs.orst.edu:80#~tadepall#research#papers#theory-of-ebl.ps',
                  'http_##www.cs.gmu.edu:80#research#gag#papers#ijcai95.ps',
                  'http_##www.swi.psy.uva.nl#usr#remco#postscripts#Straatman:95b.ps.gz',
                  'http_##www.cs.orst.edu:80#~tadepall#research#papers#tree-structured-bias-1.ps',
                  'ftp_##cns-ftp.bu.edu#pub#diana#GroMinWil:95.ps.gz',
                  'http_##mnemosyne.itc.it:1024#blanzieri#lrobot95.ps.Z',
                  'http_##www.swi.psy.uva.nl#usr#remco#postscripts#Straatman:94a.ps.gz',
                  'http_##www.cs.gmu.edu:80#research#gag#papers#gengaps.ps',
                  'http_##www.swi.psy.uva.nl#usr#remco#postscripts#Pos:97c.ps.gz',
                  'http_##mnemosyne.itc.it:1024#~avesani#papers#ida97.ps.Z',
                  'http_##www.swi.psy.uva.nl#usr#remco#postscripts#Straatman:95a.ps.gz',
                  'http_##www.cs.gmu.edu:80#research#gag#papers#fuzzy.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr253.ps',
                  'http_##mnemosyne.itc.it:1024#ricci#papers#IRST-TR-9404-07.ps.Z',
                  'http_##mnemosyne.itc.it:1024#~avesani#papers#aiia93.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr297.ps',
                  'http_##www.aic.nrl.navy.mil:80#~aha#aaai95-fss#papers#hanney.ps.Z',
                  'http_##or.eng.tau.ac.il:7777#topics#mld.ps.gz',
                  'http_##www.stat.washington.edu:80#tech.reports#tr319.ps',
                  'http_##c.gp.cs.cmu.edu:5103#afs#cs#project#sensor-9#ftp#papers#nato.ps',
                  'http_##www.cs.rice.edu:80#~cding#documents#unroll.ps.gz',
                  'http_##www.cs.gmu.edu:80#research#gag#papers#TAI93.ps',
                  'ftp_##cns-ftp.bu.edu#pub#paolo#GauZalLop:95.ps.gz',
                  'http_##mnemosyne.itc.it:1024#~avesani#papers#ewsp95.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr275.ps',
                  'http_##www.stat.washington.edu:80#tech.reports#tr332.ps',
                  'http_##www.cl.cam.ac.uk:80#ftp#papers#reports#TR275-bdp-go.ps.gz',
                  'http_##www.stat.washington.edu:80#tech.reports#tr335.ps',
                  'http_##www-cad.eecs.berkeley.edu:80#HomePages#aml#publications#nips93_def.ps.gz',
                  ]:
            ti,ab='',''
            #text.append(ti+'\n'+ab)
            title.append(ti)
            abss.append(ab)
        else:
            with open(path+fn) as f:
                lines = f.read().splitlines()
            ti, ab = '', ''
            for line in lines:
                if 'Title:' in line:
                    ti = line[7:]
                if 'Abstract:' in line:
                    ab = line[10:]
            #text.append(ti+'\n'+ab)
            title.append(ti)
            abss.append(ab)
    return data, title, abss

import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


data, title,abss = get_raw_text_cora()
save_pickle(abss,'cora_abs.pkl')     # Save Abstract Info from TAPE
save_pickle(title,'cora_title.pkl')   # Save Title Info from TAPE



# Next, we begin to align the node ID between TAPE and official Cora.
from torch_geometric.datasets import Planetoid
dataset = Planetoid('./cora_data', 'Cora',split='public')
g=dataset[0]


same=0

a={}
for j in range(2708):    # Align ID according to official numerical embedding
    tt=g.x[j]
    a[j]=[]
    for i in range(2708):
        if tt.equal(data.x[i]):
            a[j].append(i)
            same+=1
print()
print(same)  # 2752 > 2708, which means there are repeated nodes in both official Cora and TAPE Cora
#For an example, print(data.x[2150].equal(data.x[2183]))   #The 2 nodes are totally same.

print(len(a)) # 2708
print()
rest=[]
for k,v in a.items():
    if len(v)>1:
        rest.append([k,len(v),v])
save_pickle(a, 'cora_map.pkl')    # This file must be saved for next preprocessing steps
save_pickle(rest, 'cora_rest.pkl')  # We suggest saving this file for obtaining the 'repeat' list in cora_preprocess_2.py
print(rest)
print(len(rest))  # 27, which means there are 27 points with different index ID in official Cora dataset matches more than 1 point in TAPE Cora.
sum=0
all=[]
for k in rest:
    all.append(list(set(k[2])))
    sum+=k[1]-1
print(sum)   # 44 = 2752-2708
print(all)
print(len(all))  # 27 = len(rest) , which means there are no repeated nodes sharing the same index ID.
simple=[]
for thin in all:
    if thin not in simple and thin[::-1] not in simple:
        simple.append(thin)
print(len(simple))  # 11 = 7 + 3 + 1, i.e. 7 overlaps with size 2, 3 overlaps with size 3 and 1 overlap with size 4
print(simple)
print(len(rest))  # 27 = 7*2 + 3*3 + 1*4
#save_pickle(all,'cora_all.pkl')
#save_pickle(simple,'cora_simple.pkl')