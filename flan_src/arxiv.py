from fileinput import lineno
from platform import node
import re#
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
import copy


from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import P5Tokenizer, P5TokenizerFast

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

    

class Arxiv_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks    #i.e. all templates
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type

        
        print('Data sources: ', split.split(','))
        self.mode = mode
        self.prefix_1='Perform Link Prediction for the node: Node represents academic paper with a specific topic, link represents a citation between the two papers. Pay attention to the multi-hop link relationship between the nodes. '
        self.prefix_2='Categorize the article by topic: Node represents academic paper with a specific topic, link represents a citation between the two papers. Pay attention to the multi-hop link relationship between the nodes. '
        self.label_map=load_pickle(os.path.join('Arxiv','my_data','label_map.pkl'))  #1
        self.re_id=load_pickle(os.path.join('Arxiv','my_data','re_id.pkl'))  #2
        self.l_max=self.args.max_text_length
        self.real_feature=load_pickle(os.path.join('Arxiv','my_data','L_giant.pkl')) #3
        self.train_L1=load_pickle(os.path.join('Arxiv','my_data','L1.pkl'))  #4
       # self.train_L2=load_pickle(os.path.join('Arxiv','my_data','L2.pkl'))  #5  看来L2也太大了,得现场生成了
        self.transductive=load_pickle(os.path.join('Arxiv','my_data','transductive.pkl'))  #6 a list
        self.classification=load_pickle(os.path.join('Arxiv','my_data','classification.pkl'))  #7
        self.node_feature=load_pickle(os.path.join('Arxiv','my_data','node_feature.pkl'))  #8

        LA=[]
        LAA=list(set(self.label_map.values()))
        for laa in tqdm(range(len(LAA))):
            LA.append(LAA[laa])
        assert len(LA)==40 
        self.LA=LA

        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        if self.mode=='train':
            self.compute_datum_info_train()      #其实最有技术含量的在这里
        else:
            self.compute_datum_info_val()
            
        #self.total_length
        if self.mode=='val':
            self.len_transductive=len(self.transductive)   #per-template地等长
        
    def compute_datum_info_train(self):
        curr = 0
        for key in list(self.task_list.keys()):     
            if key == 'link':
                for tems in self.task_list[key]: 
                    if '1-1-1-1' in tems:
                        self.total_length += 169343 * 1  
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-1'))
                        curr = self.total_length
                    elif '1-1-2-1' in tems:  
                        self.total_length += 169343 * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-2'))
                        curr = self.total_length
                    elif '1-1-3-1' in tems:  
                        self.total_length += 169343 * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,'1-3'))  
                        curr = self.total_length

            elif key == 'classification':  # 以hop水平分组，4+8+8+(2),后面改写pretrain.py的时候要注意！！！！！
                for tems in self.task_list[key]:
                    if '2-1-1-2' in tems:
                        self.total_length += len(self.classification) * 1   #90941 nodes for training
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'2-1','transductive')) 
                            #self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'transductive'))  
                        curr = self.total_length
                    elif '2-1-2-2' in tems:

                        self.total_length += len(self.classification) * 1
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'2-2','transductive'))
                            #self.datum_info.append((i + curr, key, i // 8,tems[i % 8],'transductive'))
                        curr = self.total_length
                    elif '2-1-3-2' in tems:

                        self.total_length += len(self.classification) * 1
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'2-3','transductive'))
                            #self.datum_info.append((i + curr, key, i // 8,tems[i % 8],'transductive'))
                        curr = self.total_length
                    elif '6-6-6-6' in tems:

                        self.total_length += len(self.classification) * 1
                        for i in tqdm(range(self.total_length - curr)):
                            self.datum_info.append((i + curr, key, i // 1,'5-6','transductive'))
                            #self.datum_info.append((i + curr, key, i // 2,tems[i % 2],'transductive'))
                        curr = self.total_length
            elif key == 'intermediate':
                pass
            else:
                raise NotImplementedError

    def compute_datum_info_val(self):
        curr = 0
        for key in list(self.task_list.keys()):     
            if key == 'link':
                pass
            elif key == 'classification':
                for tems in self.task_list[key]:
                    if '2-3-1-2' in tems:
                        self.total_length += len(self.transductive) * 2
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,tems[i % 2],'transductive'))  
                        curr = self.total_length
                    elif '2-1-2-2' in tems:

                        self.total_length += len(self.transductive) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'transductive'))
                        curr = self.total_length
                    elif '2-1-3-2' in tems:

                        self.total_length += len(self.transductive) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'transductive'))
                        curr = self.total_length
                    elif '6-6-6-6' in tems:

                        self.total_length += len(self.transductive) * 1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 1,tems[i % 1],'transductive'))
                        curr = self.total_length
            elif key == 'intermediate':
                pass
            else:
                raise NotImplementedError
    
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx

        if self.mode=='train':
            if len(datum_info_idx) == 5:
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]
                #task_template = self.all_tasks[task_name][datum_info_idx[3]]

                task_template_range = datum_info_idx[3]
                if task_template_range=='2-1':
                    t_set=['2-1-1-2','2-3-1-2']
                   # t_set=['2-1-1-1','2-1-1-2','2-3-1-1','2-3-1-2']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='2-2':
                    t_set=['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4']
                    #t_set=['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4','2-3-2-1','2-3-2-2','2-3-2-3','2-3-2-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='2-3':
                    t_set=['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']
                    #t_set=['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4','2-3-3-1','2-3-3-2','2-3-3-3','2-3-3-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='5-6':
                    t_set=['6-6-6-6']
                    #t_set=['5-5-5-5','6-6-6-6']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                
                
                if task_name=='classification':
                    cate=datum_info_idx[4]
                else:
                    which_idx=datum_info_idx[4]
                    flip=0
            elif len(datum_info_idx)==4:
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]

                task_template_range = datum_info_idx[3]
                if task_template_range=='1-1':
                    t_set=['1-1-1-1','1-1-1-2','1-3-1-1','1-3-1-2']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='1-2':
                    #t_set=['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4']
                    t_set=['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4','1-3-2-1','1-3-2-2','1-3-2-3','1-3-2-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='1-3':
                    #t_set=['1-1-3-2','1-1-3-4','2-3-3-2','2-3-3-4']
                    t_set=['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4','1-3-3-1','1-3-3-2','1-3-3-3','1-3-3-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]

            else:
                raise NotImplementedError
        elif self.mode=='val': 
            if len(datum_info_idx) == 5:
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]
                task_template = self.all_tasks[task_name][datum_info_idx[3]]
                if task_name=='classification':
                    cate=datum_info_idx[4]
                else:
                    which_idx=datum_info_idx[4]
                    flip=0
            else:
                raise NotImplementedError



#
        if task_name == 'link':
            if self.mode=='train': 
                link_datum=[datum_idx]  #中心节点
            elif self.mode=='val':
                pass

            if task_template['id'] == '1-1-1-1':    #记得都要加self.prefix,  记得做re_id: self.re_id
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    rand_prob = random.random()
                   # point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                   # link_datum.append(point)
                    if rand_prob > 0.5:
                        point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                        link_datum.append(point)
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        node_list=''    
                        count=0

                        negative=random.randint(0,169342)
                        while negative in self.train_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', '<extra_id_0>')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段  #1-1-1-1
                    pass

            elif task_template['id'] == '1-1-1-2':   #记得都要加self.prefix,  记得做re_id: self.re_id
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                    link_datum.append(point)
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                        #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])
                    target_text=[self.re_id[link_datum[1]],1]
                    #target_text = task_template['target'].format(self.re_id[link_datum[1]])  

                elif self.mode=='val':   #这是测试阶段   1-1-1-2, 这里做测试要负采样，改source_text,但不要显式强调单一正确性
                    pass

            elif task_template['id'] == '1-1-2-1':
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    temp_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                                temp_L2.append(ttt)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L2[random.randint(0, len(train_L2) - 1)]
                        link_datum.extend(points)
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露   
                        node_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>','<extra_id_0>')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段    #1-1-2-1
                    pass



            elif task_template['id'] == '1-1-2-2':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[train_L2[idx][1]])
                       # node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])

                    target_text = [self.re_id[link_datum[2]],1]

                elif self.mode=='val':   #这是测试阶段  1-1-2-2
                    pass


            elif task_template['id'] == '1-1-2-3':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L2[idx][1]])
                            #middle_list=middle_list+'{}, '.format(self.re_id[train_L2[idx][0]])
                            middle_list=middle_list+'<extra_id_0>, '
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>','<extra_id_0>','<extra_id_0>')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L2=self.train_L1[link_datum[1]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L2[idx][1]])
                            #middle_list=middle_list+'{}, '.format(self.re_id[train_L2[idx][0]])
                            middle_list=middle_list+'<extra_id_0>, '
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','<extra_id_0>')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-1-2-3
                    pass

            elif task_template['id'] == '1-1-2-4':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>','<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])
                        node_list=node_list+'<extra_id_0>, '
                        id_1.append(self.re_id[train_L2[idx][1]])
                        #middle_list=middle_list+'{}, '.format(self.re_id[train_L2[idx][0]])
                        middle_list=middle_list+'<extra_id_0>, '
                        id_2.append(self.re_id[train_L2[idx][0]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>','<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        id_1.pop(-1)
                        id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[link_datum[0]])
                    real_id.append(self.re_id[link_datum[1]])

                    target_text =[self.re_id[link_datum[2]],1]

                elif self.mode=='val':   #这是测试阶段  1-1-2-4
                    pass

            elif task_template['id'] == '1-1-3-1':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    temp_L3=[]
                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])
                                temp_L3.append(el)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L3[random.randint(0, len(train_L3) - 1)]
                        link_datum.extend(points)
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>','<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else: 

                        node_list=''    
                        count=0
                        negative=random.randint(0,169342)                                               
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', '<extra_id_0>')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])

                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-1-3-1
                    pass

            elif task_template['id'] == '1-1-3-2':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[train_L3[idx][2]])
                        #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])

                    target_text = [self.re_id[link_datum[3]],1]


                elif self.mode=='val':   #这是测试阶段  1-1-3-2   答案不唯一
                    pass

            elif task_template['id'] == '1-1-3-3':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])
                            #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L3=self.train_L1[link_datum[2]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])
                            #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>', '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段      1-1-3-3
                    pass
            elif task_template['id'] == '1-1-3-4': 
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        id_1.append(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                        id_2.append(self.re_id[train_L3[idx][0]])
                        id_2.append(self.re_id[train_L3[idx][1]])
                        #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>','(<extra_id_0>,<extra_id_0>)')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        id_1.pop(-1)
                        id_2.pop(-1)
                        id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[link_datum[0]])
                    real_id.append(self.re_id[link_datum[1]])
                    real_id.append(self.re_id[link_datum[2]])

                    target_text =[self.re_id[link_datum[3]],1]

                elif self.mode=='val':   #这是测试阶段   1-1-3-4
                    pass
            #








            elif task_template['id'] == '1-3-1-1':    #记得都要加self.prefix,  记得做re_id: self.re_id
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    rand_prob = random.random()
                   # point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                   # link_datum.append(point)
                    if rand_prob > 0.5:
                        point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                        link_datum.append(point)
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[1]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[link_datum[0]][idx]][0])
                            real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[1]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        node_list=''    
                        count=0

                        negative=random.randint(0,169342)
                        while negative in self.train_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[link_datum[0]][idx]][0])
                            real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   
                    pass

            elif task_template['id'] == '1-3-1-2':   #记得都要加self.prefix,  记得做re_id: self.re_id
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    point=self.train_L1[link_datum[0]][random.randint(0, len(self.train_L1[link_datum[0]]) - 1)]
                    link_datum.append(point)
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[link_datum[0]][idx]][0])
                        real_id.append(self.re_id[self.train_L1[link_datum[0]][idx]])
                        #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])
                    target_text=[self.re_id[link_datum[1]],1]
                    #target_text = task_template['target'].format(self.re_id[link_datum[1]])  

                elif self.mode=='val':   #这是测试阶段   1-1-1-2, 这里做测试要负采样，改source_text,但不要显式强调单一正确性
                    pass








            elif task_template['id'] == '1-3-2-1':
                
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    temp_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                                temp_L2.append(ttt)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L2[random.randint(0, len(train_L2) - 1)]
                        link_datum.extend(points)
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            real_id.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露   
                        node_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            real_id.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass



            elif task_template['id'] == '1-3-2-2':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                        real_id.append(self.re_id[train_L2[idx][1]])
                       # node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])

                    target_text = [self.re_id[link_datum[2]],1]

                elif self.mode=='val':   
                    pass


            elif task_template['id'] == '1-3-2-3':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            id_1.append(self.re_id[train_L2[idx][1]])
                            #middle_list=middle_list+'{}, '.format(self.re_id[train_L2[idx][0]])
                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[2]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[2]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L2=self.train_L1[link_datum[1]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            id_1.append(self.re_id[train_L2[idx][1]])
                            #middle_list=middle_list+'{}, '.format(self.re_id[train_L2[idx][0]])
                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            id_2.append(self.re_id[train_L2[idx][0]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-2-4':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    points=train_L2[random.randint(0, len(train_L2) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        #node_list=node_list+'{}, '.format(self.re_id[train_L2[idx][1]])
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                        id_1.append(self.re_id[train_L2[idx][1]])
                        #middle_list=middle_list+'{}, '.format(self.re_id[train_L2[idx][0]])
                        middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                        id_2.append(self.re_id[train_L2[idx][0]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        id_1.pop(-1)
                        id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[link_datum[0]])
                    real_id.append(self.re_id[link_datum[1]])

                    target_text =[self.re_id[link_datum[2]],1]

                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-3-1':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])
                    temp_L3=[]
                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])
                                temp_L3.append(el)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        points=train_L3[random.randint(0, len(train_L3) - 1)]
                        link_datum.extend(points)
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            real_id.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        target_text = task_template['target'].format('yes')

                    else: 

                        node_list=''    
                        count=0
                        negative=random.randint(0,169342)                                               
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            real_id.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[negative][0],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])

                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass

            elif task_template['id'] == '1-3-3-2':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                        real_id.append(self.re_id[train_L3[idx][2]])
                        #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], '<extra_id_0>',self.node_feature[link_datum[0]][0])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)
                    real_id.append(self.re_id[link_datum[0]])

                    target_text = [self.re_id[link_datum[3]],1]


                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-3-3':
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            id_1.append(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])
                            #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[3]][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[link_datum[3]])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L3=self.train_L1[link_datum[2]]
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,169342)
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,169342)

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            id_1.append(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])
                            #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2], middle_list[:-2],'<extra_id_0>',self.node_feature[negative][0], '<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[negative])
                        real_id.append(self.re_id[link_datum[0]])
                        real_id.append(self.re_id[link_datum[1]])
                        real_id.append(self.re_id[link_datum[2]])


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '1-3-3-4': 
                if self.mode=='train': 
                    real_id=[self.re_id[link_datum[0]]]
                    id_1,id_2=[],[]
                    train_L2=[]
                    for eee in self.train_L1[link_datum[0]]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=link_datum[0]:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])

                    points=train_L3[random.randint(0, len(train_L3) - 1)]
                    link_datum.extend(points)

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                        id_1.append(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                        id_2.append(self.re_id[train_L3[idx][0]])
                        id_2.append(self.re_id[train_L3[idx][1]])
                        #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        source_text =self.prefix_1 + task_template['source'].format('<extra_id_0>',self.node_feature[link_datum[0]][0], node_list[:-2],middle_list[:-2],'<extra_id_0>',self.node_feature[link_datum[0]][0],'<extra_id_0>',self.node_feature[link_datum[1]][0],'<extra_id_0>',self.node_feature[link_datum[2]][0])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        id_1.pop(-1)
                        id_2.pop(-1)
                        id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[link_datum[0]])
                    real_id.append(self.re_id[link_datum[1]])
                    real_id.append(self.re_id[link_datum[2]])

                    target_text =[self.re_id[link_datum[3]],1]

                elif self.mode=='val':   
                    pass



        
        elif task_name == 'classification':
            if self.mode=='train':   
                point=self.classification[datum_idx]
            elif self.mode=='val':
                if cate=='inductive':
                    pass
                    #point=self.inductive[datum_idx]   #实际上inductive这里这个point根本不能用,因为我都是给'A new node'
                elif cate=='transductive':
                    point=self.transductive[datum_idx]

            #统一进行label映射
            label=self.label_map[point]
            #LA=['numerical analysis','multimedia','logic','society','security','distributed computing','human computer interaction','computational engineering','internet','complexity','']
            negative=str(np.random.choice(list(set(self.LA).difference({label})),1,replace=False)[0])

            tit=self.node_feature[point][0]

            if task_template['id'] == '5-5-5-5':   #无论训练测试都一个模板pipeline
                abs=self.node_feature[point][1] 
                rand_prob=random.random()
                if rand_prob>0.5:
                    source_text =task_template['source'].format('<extra_id_0>', abs, '<extra_id_0>', label)
                    real_id=[self.re_id[point],self.re_id[point]]
                    target_text = task_template['target'].format('yes')
                else:
                    source_text =task_template['source'].format('<extra_id_0>', abs, '<extra_id_0>', negative)
                    real_id=[self.re_id[point],self.re_id[point]]
                    target_text = task_template['target'].format('no')
#
            elif task_template['id']=='6-6-6-6':
                abs=self.node_feature[point][1] 
                source_text =task_template['source'].format('<extra_id_0>', abs, '<extra_id_0>')
                real_id=[self.re_id[point],self.re_id[point]]   #直接当成tokenize过了
                target_text = task_template['target'].format(label)
#

            elif task_template['id'] == '2-1-1-1':
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[self.train_L1[point][idx]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了

                        real_id.append(self.re_id[point])     #记得append这一下
                        target_text = task_template['target'].format('yes')

                    else:     #对分类类别进行负label采样
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[self.train_L1[point][idx]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')

                elif self.mode=='val':   #这是测试阶段  #2-1-1-1
                    pass


            elif task_template['id'] == '2-1-1-2':
                if self.mode!=None: 
                    #
                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[self.train_L1[point][idx]])
                        #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   #2-1-1-2
                    pass
            
            elif task_template['id'] == '2-1-2-1':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L2[idx][1]])

                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': #2-1-2-1
                    pass
            elif task_template['id'] == '2-1-2-2':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[train_L2[idx][1]])
                        #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': #2-1-2-2     #!!!!!!!
                    pass
            elif task_template['id'] == '2-1-2-3':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'<extra_id_0>, '
                            id_2.append(self.re_id[train_L2[idx][0]])
                            #middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>', label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'<extra_id_0>, '
                            id_2.append(self.re_id[train_L2[idx][0]])
                            #middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        real_id=real_id+id_1+id_2

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': #2-1-2-3
                    pass
            elif task_template['id'] == '2-1-2-4':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        id_1.append(self.re_id[train_L2[idx][1]])
                        #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                        middle_list=middle_list+'<extra_id_0>, '
                        id_2.append(self.re_id[train_L2[idx][0]])
                        #middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)

                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  #2-1-2-4
                    pass




            elif task_template['id'] == '2-1-3-1':    #3阶还是有点特殊的
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        #train_L3=[]   #这个用作node采样
                        #for ele in train_L2:
                         #   for el in self.train_L1[ele[1]]:
                          #      if el!=ele[0] and el!=point:
                           #         train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>', label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        real_id=[self.re_id[point]]

                        #train_L3=[]   #这个用作node采样
                        #for ele in train_L2:
                         #   for el in self.train_L1[ele[1]]:
                          #      if el!=ele[0] and el!=point:
                           #         train_L3.append(ele+[el])
                        

                        node_list=''    
                        count=0
                        

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            real_id.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], '<extra_id_0>', negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   2-1-3-1
                    pass
            elif task_template['id'] == '2-1-3-2':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]

                    #train_L3=[]   #这个用作node采样
                    #for ele in train_L2:
                     #   for el in self.train_L1[ele[1]]:
                      #      if el!=ele[0] and el!=point:
                       #         train_L3.append(ele+[el])
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'<extra_id_0>, '
                        real_id.append(self.re_id[train_L3[idx][2]])
                        #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],'<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)


                elif self.mode=='val':  #2-1-3-2
                    pass

            elif task_template['id'] == '2-1-3-3':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        #train_L3=[]   #这个用作node采样
                        #for ele in train_L2:
                         #   for el in self.train_L1[ele[1]]:
                          #      if el!=ele[0] and el!=point:
                           #         train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>',label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])
                            #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2], middle_list[:-2],'<extra_id_0>',label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        #train_L3=[]   #这个用作node采样
                        #for ele in train_L2:
                         #   for el in self.train_L1[ele[1]]:
                          #      if el!=ele[0] and el!=point:
                           #         train_L3.append(ele+[el])
                      
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>',negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            node_list=node_list+'<extra_id_0>, '
                            id_1.append(self.re_id[train_L3[idx][2]])

                            middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])
                            #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>', negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段      2-1-3-3
                    pass
            elif task_template['id'] == '2-1-3-4':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    #train_L3=[]   #这个用作node采样
                    #for ele in train_L2:
                     #   for el in self.train_L1[ele[1]]:
                      #      if el!=ele[0] and el!=point:
                       #         train_L3.append(ele+[el])
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2],'<extra_id_0>')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        node_list=node_list+'<extra_id_0>, '
                        id_1.append(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'(<extra_id_0>,<extra_id_0>), '
                        id_2.append(self.re_id[train_L3[idx][0]])
                        id_2.append(self.re_id[train_L3[idx][1]])
                        #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>', node_list[:-2],middle_list[:-2], '<extra_id_0>')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   #这是测试阶段   2-1-3-4
                    pass
            


            elif task_template['id'] == '2-3-1-1':    #while循环一定会进去
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                            real_id.append(self.re_id[self.train_L1[point][idx]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了

                        real_id.append(self.re_id[point])     #记得append这一下
                        target_text = task_template['target'].format('yes')

                    else:     #对分类类别进行负label采样
                        real_id=[self.re_id[point]]

                        node_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                            real_id.append(self.re_id[self.train_L1[point][idx]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')

                elif self.mode=='val':  
                    pass


            elif task_template['id'] == '2-3-1-2':
                if self.mode!=None: 
                    #
                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[self.train_L1[point][idx]][0])
                        real_id.append(self.re_id[self.train_L1[point][idx]])
                        #node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit)
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)

                elif self.mode=='val': 
                    pass
            
            elif task_template['id'] == '2-3-2-1':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            real_id.append(self.re_id[train_L2[idx][1]])

                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        real_id=[self.re_id[point]]
                        node_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            real_id.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-2':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                        real_id.append(self.re_id[train_L2[idx][1]])
                        #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit)
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-3':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            id_1.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            id_2.append(self.re_id[train_L2[idx][0]])
                            #middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit, label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                            id_1.append(self.re_id[train_L2[idx][1]])
                            #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                            id_2.append(self.re_id[train_L2[idx][0]])
                            #middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                        real_id=real_id+id_1+id_2

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val': 
                    pass
            elif task_template['id'] == '2-3-2-4':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L2):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L2)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][1]][0])
                        id_1.append(self.re_id[train_L2[idx][1]])
                        #node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                        middle_list=middle_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L2[idx][0]][0])
                        id_2.append(self.re_id[train_L2[idx][0]])
                        #middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit)
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)

                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  
                    pass




            elif task_template['id'] == '2-3-3-1':    #3阶还是有点特殊的
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])
                    
                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]

                        #train_L3=[]   #这个用作node采样
                        #for ele in train_L2:
                         #   for el in self.train_L1[ele[1]]:
                          #      if el!=ele[0] and el!=point:
                           #         train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            real_id.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit, label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        real_id=[self.re_id[point]]

                        #train_L3=[]   #这个用作node采样
                        #for ele in train_L2:
                        #    for el in self.train_L1[ele[1]]:
                         #       if el!=ele[0] and el!=point:
                          #          train_L3.append(ele+[el])
                        

                        node_list=''    
                        count=0
                        

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            real_id.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], '<extra_id_0>',tit, negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            real_id.pop(-1)

                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass
            elif task_template['id'] == '2-3-3-2':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])
                    
                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]

                   # train_L3=[]   #这个用作node采样
                    #for ele in train_L2:
                     #   for el in self.train_L1[ele[1]]:
                      #      if el!=ele[0] and el!=point:
                       #         train_L3.append(ele+[el])
                    node_list=''    
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                        real_id.append(self.re_id[train_L3[idx][2]])
                        #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],'<extra_id_0>',tit)
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        real_id.pop(-1)

                    real_id.append(self.re_id[point])
                    target_text = task_template['target'].format(label)


                elif self.mode=='val':  
                    pass

            elif task_template['id'] == '2-3-3-3':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                       # train_L3=[]   #这个用作node采样
                       # for ele in train_L2:
                        #    for el in self.train_L1[ele[1]]:
                         #       if el!=ele[0] and el!=point:
                          #          train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit,label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            id_1.append(self.re_id[train_L3[idx][2]])
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])
                            #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2], middle_list[:-2],'<extra_id_0>',tit,label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)
                        #我要做到当while结束的时候source_text已经ok了
                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        real_id=[self.re_id[point]]
                        id_1,id_2=[],[]

                       # train_L3=[]   #这个用作node采样
                        #for ele in train_L2:
                         #   for el in self.train_L1[ele[1]]:
                          #      if el!=ele[0] and el!=point:
                           #         train_L3.append(ele+[el])
                      
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit,negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                            id_1.append(self.re_id[train_L3[idx][2]])

                            middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                            id_2.append(self.re_id[train_L3[idx][0]])
                            id_2.append(self.re_id[train_L3[idx][1]])
                            #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit, negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                            if len(id_1)>0:
                                id_1.pop(-1)
                                id_2.pop(-1)
                                id_2.pop(-1)

                        real_id=real_id+id_1+id_2
                        real_id.append(self.re_id[point])
                        target_text = task_template['target'].format('no')
                elif self.mode=='val':  
                    pass
            elif task_template['id'] == '2-3-3-4':
                if self.mode!=None: 
                    train_L2=[]
                    for eee in self.train_L1[point]:
                        for ttt in self.train_L1[eee]:
                            if ttt!=point:
                                train_L2.append([eee,ttt])

                    train_L3=[]   #这个用作node采样
                    random.shuffle(train_L2)
                    for ele in train_L2[:200]:
                        ta=copy.deepcopy(self.train_L1[ele[1]])
                        random.shuffle(ta)
                        for el in ta[:30]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])

                    real_id=[self.re_id[point]]
                    id_1,id_2=[],[]

                    #train_L3=[]   #这个用作node采样
                    #for ele in train_L2:
                     #   for el in self.train_L1[ele[1]]:
                      #      if el!=ele[0] and el!=point:
                       #         train_L3.append(ele+[el])
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2],'<extra_id_0>',tit)
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        #node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        node_list=node_list+'(<extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][2]][0])
                        id_1.append(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'(<extra_id_0>,{}; <extra_id_0>,{}), '.format(self.node_feature[train_L3[idx][0]][0],self.node_feature[train_L3[idx][1]][0])
                        id_2.append(self.re_id[train_L3[idx][0]])
                        id_2.append(self.re_id[train_L3[idx][1]])
                        #middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        source_text =self.prefix_2 + task_template['source'].format('<extra_id_0>',tit, node_list[:-2],middle_list[:-2], '<extra_id_0>',tit)
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        if len(id_1)>0:
                            id_1.pop(-1)
                            id_2.pop(-1)
                            id_2.pop(-1)
                    real_id=real_id+id_1+id_2
                    real_id.append(self.re_id[point])

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  
                    pass

            else:
                raise NotImplementedError
            

        elif task_name == 'intermediate':  #暂时没加
            pass
        else:
            raise NotImplementedError
            

        input_ids = self.tokenizer.encode(source_text, padding=True, truncation=True, max_length=512)
        extra_num=0
        for idi in range(len(input_ids)):
            idid=input_ids[idi]
            if idid==32099:
                input_ids[idi]=real_id[extra_num]
                extra_num+=1
        if extra_num!=len(real_id):
            print(task_template['id'])
            print(source_text)
            print(extra_num,len(real_id))
        assert extra_num==len(real_id)

                
        #tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids=[0]*len(input_ids) #反正我也不用
        #whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
       # assert len(whole_word_ids) == len(input_ids)
        if task_template['id'].startswith('1') and (task_template['id'].endswith('2') or task_template['id'].endswith('4')):
            target_ids=target_text
        else:
            target_ids = self.tokenizer.encode(target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
       # out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight
        out_dict['temp_id'] = task_template['id']

        if self.mode=='val':
            out_dict['cate']='None' if task_template['task']!='classification' else cate

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] ## the added [0] is for </s>
    
    def collate_fn(self, batch):   #Most important， this will call after the 'get_item_'
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []
        temp_ids=[]
        cate=[]

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

            if 'temp_id' in entry:
                temp_ids.append(entry['temp_id'])

            if 'cate' in entry:
                cate.append(entry['cate'])
            

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights
        batch_entry['temp_ids'] = temp_ids   #这个我自己加的, collate_fn我没有整体重写,but I essentially changed it
        if len(cate)!=0:
            batch_entry['cate'] = cate

        return batch_entry     # Real return, i.e. real batch datas , 目前没有把tokenized_text传出去
