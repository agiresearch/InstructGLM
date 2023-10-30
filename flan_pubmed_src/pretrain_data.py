from operator import neg
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

    

    
#为什么P5里面需要两套template呢？这个问题我的初步想法是：为了区别在amazon中是购买记录而yelp中是浏览记录。实际上属于edge_type
#就是要说明‘link’的内涵
class beauty_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks    #i.e. all templates
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type

        self.item_l=12101
        self.user_l=22363
        
        print('Data sources: ', split.split(','))
        self.mode = mode
        self.prefix='node represents user or item, link represents purchase behaviour with review. '
        self.edge_feature=load_pickle(os.path.join('my_data','edge_feature.pkl'))
        self.label_map=load_pickle(os.path.join('my_data','label_map.pkl'))
        self.isolated=load_pickle(os.path.join('my_data','isolated.pkl'))  #这个只需要在classification测试的时候用到，其他时候都采不到, 负例采样的时候也用得到
        self.re_id=load_pickle(os.path.join('my_data','re_id.pkl'))
        self.l_max=self.args.max_text_length


        if self.mode == 'train':    
            self.d_inductive=load_pickle(os.path.join('my_data','d_inductive.pkl'))
            inductive=[]
            for kk in self.d_inductive.keys():
                inductive=inductive+list(self.d_inductive[kk])
            self.inductive=inductive

            self.train_classification=load_pickle(os.path.join('my_data','train_classification.pkl'))
            self.train_L1=load_pickle(os.path.join('my_data','train_L1.pkl'))   # 3hop信息要现场索
            self.train_L2=load_pickle(os.path.join('my_data','train_L2.pkl'))
            self.L1=load_pickle(os.path.join('my_data','L1.pkl'))
            self.L2=load_pickle(os.path.join('my_data','L2.pkl'))

            train_p=load_pickle(os.path.join('my_data','train_path.pkl'))
            train_path=list(train_p.values())
            self.train_path=train_path

            classification=[]
            for kk in self.train_classification.keys():
                classification=classification+self.train_classification[kk]
            self.classification=classification
        elif self.mode == 'val':
            self.d_inductive=load_pickle(os.path.join('my_data','d_inductive.pkl'))
            self.d_transductive=load_pickle(os.path.join('my_data','d_transductive.pkl'))
            self.L1=load_pickle(os.path.join('my_data','L1.pkl'))
            self.L2=load_pickle(os.path.join('my_data','L2.pkl'))

            test_p=load_pickle(os.path.join('my_data','test_path.pkl'))
            test_path=list(test_p.values())
            self.test_path=test_path

            self.t_L1=load_pickle(os.path.join('my_data','t_L1.pkl'))
            self.t_L2=load_pickle(os.path.join('my_data','t_L2.pkl'))
            inductive=[]
            for kk in self.d_inductive.keys():
                inductive=inductive+list(self.d_inductive[kk])
            self.inductive=inductive

            transductive=[]
            for kk in self.d_transductive.keys():
                transductive=transductive+list(self.d_transductive[kk])   #注意这里要list转换
            self.transductive=transductive

            #self.test_classification=inductive+transductive

        else:
            raise NotImplementedError
            
        print('compute_datum_info')#
        self.total_length = 0
        self.datum_info = []
        if self.mode=='train':
            self.compute_datum_info_train()
        else:
            self.compute_datum_info_val()
        #self.total_length
        if self.mode=='val':
            len_link=defaultdict(int)
            len_transductive=defaultdict(int)
            len_inductive=defaultdict(int)
            for e in self.datum_info:
                if e[1] == 'classification':
                    if e[4]=='inductive':
                        len_inductive[e[3]]+=1
                    else:
                        len_transductive[e[3]]+=1
                else:
                    len_link[e[3]]+=1
            self.len_link=len_link
            self.len_transductive=len_transductive
            self.len_inductive=len_inductive
        
        #记得回来这里统计test情况下各个template对应的子数据集长度
        
    def compute_datum_info_train(self):
        curr = 0
        for key in list(self.task_list.keys()):     
            if key == 'link':
                for tems in self.task_list[key]:  # 以hop水平分组，4+8+8,history不用了
                    if '1-1-1-1' in tems:
                        self.total_length += len(self.train_path) * 2
                        flip=0 #顺序
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,'1-1',flip))   #len=5
                            flip=1-flip
                        curr = self.total_length
                    elif '1-1-2-1' in tems:
                        self.total_length += len(self.train_path) * 2
                        which=1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,'1-2',which))
                            which=3-which
                        curr = self.total_length
                    elif '1-1-3-1' in tems:
                        self.total_length += len(self.train_path) * 3
                        which=3
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 3,'1-3',which))
                            which=int(-1.5*which**2+11.5*which-17)   #3,4,5的轮回
                        curr = self.total_length
            elif key == 'classification':
                for tems in self.task_list[key]:
                    if '2-1-1-1' in tems:
                        self.total_length += len(self.classification) * 2
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,'2-1'))   
                        curr = self.total_length
                    elif '2-1-2-1' in tems:
                        self.total_length += len(self.classification) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,'2-2'))
                        curr = self.total_length
                    elif '2-1-3-1' in tems:
                        self.total_length += len(self.classification) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,'2-3'))
                        curr = self.total_length
            elif key == 'intermediate':
                for tems in self.task_list[key]:
                    if '3-1-1-1' in tems:
                        self.total_length += len(self.train_path) * 2   #给1hop， 问2hop
                        which=1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,'3-1',which))   
                            which=3-which
                        curr = self.total_length
                    elif '3-1-2-1' in tems:
                        self.total_length += len(self.train_path) * 2
                        which=1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,'3-2',which))
                            which=3-which
                        curr = self.total_length
                    elif '3-1-3-1' in tems:
                        self.total_length += len(self.train_path) * 3
                        which=3
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 3,'3-3',which))
                            which=int(-1.5*which**2+11.5*which-17) 
                        curr = self.total_length
            else:
                raise NotImplementedError

    def compute_datum_info_val(self):
        curr = 0
        for key in list(self.task_list.keys()):     
            if key == 'link':
                for tems in self.task_list[key]:  # 以hop水平分组，4+8+8,history不用了, 后面改写pretrain.py的时候要注意！！！！！
                    if '1-1-1-1' in tems:
                        self.total_length += len(self.test_path) * 4
                        which=0 #顺序
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],which))   
                        curr = self.total_length
                    elif '1-1-2-1' in tems:
                        self.total_length += len(self.test_path) * 8
                        which=1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 8,tems[i % 8],which))
                        curr = self.total_length
                    elif '1-1-3-1' in tems:
                        self.total_length += len(self.test_path) * 8
                        which=5
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 8,tems[i % 8],which))
                        curr = self.total_length
            elif key == 'classification':
                for tems in self.task_list[key]:
                    if '2-1-1-1' in tems:
                        self.total_length += len(self.inductive) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'inductive'))  
                        curr = self.total_length

                        self.total_length += len(self.transductive) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'transductive'))  
                        curr = self.total_length
                    elif '2-1-2-1' in tems:
                        self.total_length += len(self.inductive) * 8
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 8,tems[i % 8],'inductive'))
                        curr = self.total_length

                        self.total_length += len(self.transductive) * 8
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 8,tems[i % 8],'transductive'))
                        curr = self.total_length
                    elif '2-1-3-1' in tems:
                        self.total_length += len(self.inductive) * 8
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 8,tems[i % 8],'inductive'))
                        curr = self.total_length

                        self.total_length += len(self.transductive) * 8
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 8,tems[i % 8],'transductive'))
                        curr = self.total_length
            elif key == 'intermediate':
                for tems in self.task_list[key]:
                    if '3-1-1-1' in tems:
                        self.total_length += len(self.test_path) * 4   #给1hop， 问2hop
                        which=1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],which))   
                        curr = self.total_length
                    elif '3-1-2-1' in tems:
                        self.total_length += len(self.test_path) * 8
                        which=1
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 8,tems[i % 8],which))
                        curr = self.total_length
                    elif '3-1-3-1' in tems:
                        self.total_length += len(self.test_path) * 8
                        which=4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 8,tems[i % 8],which)) 
                        curr = self.total_length
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
                task_template_range = datum_info_idx[3]
                if task_template_range=='1-1':
                    flip = datum_info_idx[4]
                    which_idx=0
                    t_set=['1-1-1-1','1-1-1-2','1-2-1-1','1-2-1-2']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                else:
                    which_idx=datum_info_idx[4] 
                    flip=random.randint(0,1)
                    if task_template_range=='1-2':
                        t_set=['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4','1-2-2-1','1-2-2-2','1-2-2-3','1-2-2-4']
                        task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                    if task_template_range=='1-3':
                        t_set=['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4','1-2-3-1','1-2-3-2','1-2-3-3','1-2-3-4']
                        task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                    if task_template_range=='3-1':
                        t_set=['3-1-1-1','3-1-1-2','3-2-1-1','3-2-1-2']
                        task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                    if task_template_range=='3-2':
                        t_set=['3-1-2-1','3-1-2-2','3-1-2-3','3-1-2-4','3-2-2-1','3-2-2-2','3-2-2-3','3-2-2-4']
                        task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                    if task_template_range=='3-3':
                        t_set=['3-1-3-1','3-1-3-2','3-1-3-3','3-1-3-4','3-2-3-1','3-2-3-2','3-2-3-3','3-2-3-4']
                        task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]

            elif len(datum_info_idx) == 4:   #必是classification
                task_name = datum_info_idx[1]
                datum_idx = datum_info_idx[2]
                task_template_range = datum_info_idx[3]
                if task_template_range=='2-1':
                    t_set=['2-1-1-1','2-1-1-2','2-2-1-1','2-2-1-2']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='2-2':
                    t_set=['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4','2-2-2-1','2-2-2-2','2-2-2-3','2-2-2-4']
                    task_template=self.all_tasks[task_name][t_set[random.randint(0,len(t_set)-1)]]
                if task_template_range=='2-3':
                    t_set=['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4','2-2-3-1','2-2-3-2','2-2-3-3','2-2-3-4']
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

        if task_name == 'link':
            if self.mode=='train':   #根据which_idx知道edge的类型后，还需要根据which_idx进一步得到缺失边的索引信息? 好像无需
                if flip==0:
                    link_datum = self.train_path[datum_idx][which_idx]
                else:
                    link_datum = self.train_path[datum_idx][which_idx][::-1]
            elif self.mode=='val':
                link_datum = self.test_path[datum_idx][which_idx] #val时候flip必为0
            
            #下面建立最长512采样机制########################
            #template要分'train','val'讨论

            if task_template['id'] == '1-1-1-1':    #记得都要加self.prefix,  记得做re_id: self.re_id
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[1]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[1]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        node_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in self.L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段  #1-1-1-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[1]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L1[link_datum[0]][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[1]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        node_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in self.t_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L1[link_datum[0]][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')

            elif task_template['id'] == '1-1-1-2':   #记得都要加self.prefix,  记得做re_id: self.re_id
                if self.mode=='train': 
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[1]])

                elif self.mode=='val':   #这是测试阶段   1-1-1-2, 这里做测试要负采样，改source_text,但不要显式强调单一正确性
                    #
                    candidate = []
                    candidate_num = 9
                    while len(candidate) < candidate_num:
                        sample_ids = np.random.choice(list(range(1,self.item_l+self.user_l+1)), candidate_num, replace=False)
                        sample_ids = [str(self.re_id[item]) for item in sample_ids if item not in self.isolated and item not in self.inductive and item not in self.t_L1[link_datum[0]] and item!=link_datum[0] and str(self.re_id[item]) not in candidate]
                        candidate.extend(sample_ids)
                    candidate = candidate[:candidate_num]
                    candidate.append(str(self.re_id[link_datum[1]]))
                    random.shuffle(candidate)
                    post=' \n Pick the most suitable node from the following list: {}.'.format(', '.join(candidate))
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])+post
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.L1[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.L1[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.L1[link_datum[0]][idx]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])+post
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[1]])


            elif task_template['id'] == '1-2-1-1':   #要用到self.edge_feature了
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx=(link_datum[0],link_datum[1])
                        else:
                            f_dx=(link_datum[1],link_datum[0])
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1], self.re_id[link_datum[1]], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])
                            add=(link_datum[0],self.train_L1[link_datum[0]][idx]) if link_datum[0]>self.train_L1[link_datum[0]][idx] else (self.train_L1[link_datum[0]][idx],link_datum[0])
                            feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[link_datum[1]], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        node_list=''    
                        feature_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx=(link_datum[0],link_datum[1])
                        else:
                            f_dx=(link_datum[1],link_datum[0])
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in self.L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])
                            add=(link_datum[0],self.train_L1[link_datum[0]][idx]) if link_datum[0]>self.train_L1[link_datum[0]][idx] else (self.train_L1[link_datum[0]][idx],link_datum[0])
                            feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-2-1-1    
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx=(link_datum[0],link_datum[1])
                        else:
                            f_dx=(link_datum[1],link_datum[0])
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1], self.re_id[link_datum[1]], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.L1[link_datum[0]][idx]])
                            add=(link_datum[0],self.L1[link_datum[0]][idx]) if link_datum[0]>self.L1[link_datum[0]][idx] else (self.L1[link_datum[0]][idx],link_datum[0])
                            feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[link_datum[1]], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')


                    
                    else:  #得做负例采样
                        node_list=''    
                        feature_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx=(link_datum[0],link_datum[1])
                        else:
                            f_dx=(link_datum[1],link_datum[0])
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in self.t_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L1[link_datum[0]][idx]])
                            add=(link_datum[0],self.L1[link_datum[0]][idx]) if link_datum[0]>self.L1[link_datum[0]][idx] else (self.L1[link_datum[0]][idx],link_datum[0])
                            feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')

            elif task_template['id'] == '1-2-1-2':
                if self.mode=='train': 
                    #
                    node_list=''    
                    feature_list=''
                    if link_datum[0]>link_datum[1]:
                        f_dx=(link_datum[0],link_datum[1])
                    else:
                        f_dx=(link_datum[1],link_datum[0])
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])
                        add=(link_datum[0],self.train_L1[link_datum[0]][idx]) if link_datum[0]>self.train_L1[link_datum[0]][idx] else (self.train_L1[link_datum[0]][idx],link_datum[0])
                        feature_list=feature_list + '\'{}\'; '.format(self.edge_feature[add])   #

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[1]])

                elif self.mode=='val':     #1-2-1-2, 答案是唯一的
                    node_list=''    
                    feature_list=''
                    if link_datum[0]>link_datum[1]:
                        f_dx=(link_datum[0],link_datum[1])
                    else:
                        f_dx=(link_datum[1],link_datum[0])
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.L1[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.L1[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.L1[link_datum[0]][idx]])
                        add=(link_datum[0],self.L1[link_datum[0]][idx]) if link_datum[0]>self.L1[link_datum[0]][idx] else (self.L1[link_datum[0]][idx],link_datum[0])
                        feature_list=feature_list + '\'{}\'; '.format(self.edge_feature[add])   #

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx]+'\';')
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[1]])

            elif task_template['id'] == '1-1-2-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[2]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[2]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段    #1-1-2-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[2]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[2]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.t_L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')



            elif task_template['id'] == '1-1-2-2':
                if self.mode=='train': 
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])

                elif self.mode=='val':   #这是测试阶段  1-1-2-2
                    temp_L2=[]   #这个操作保持顺序
                    for ele in self.t_L2[link_datum[0]]:
                        temp_L2.append(ele[1])
                    #
                    candidate = []
                    candidate_num = 9
                    while len(candidate) < candidate_num:
                        sample_ids = np.random.choice(list(range(1,self.item_l+self.user_l+1)), candidate_num, replace=False)
                        sample_ids = [str(self.re_id[item]) for item in sample_ids if item not in self.isolated and item not in self.inductive and item not in temp_L2 and item!=link_datum[0] and str(self.re_id[item]) not in candidate]
                        candidate.extend(sample_ids)
                    candidate = candidate[:candidate_num]
                    candidate.append(str(self.re_id[link_datum[2]]))
                    random.shuffle(candidate)
                    post=' \n Pick the most suitable node from the following list: {}.'.format(', '.join(candidate))
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])+post
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])+post
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])


            elif task_template['id'] == '1-1-2-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-1-2-3
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.t_L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')

            elif task_template['id'] == '1-1-2-4':
                if self.mode=='train': 
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                        middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])

                elif self.mode=='val':   #这是测试阶段  1-1-2-4
                    temp_L2=[]   #这个操作保持顺序
                    for ele in self.t_L2[link_datum[0]]:
                        temp_L2.append(ele[1])
                    #
                    candidate = []
                    candidate_num = 9
                    while len(candidate) < candidate_num:
                        sample_ids = np.random.choice(list(range(1,self.item_l+self.user_l+1)), candidate_num, replace=False)
                        sample_ids = [str(self.re_id[item]) for item in sample_ids if item not in self.isolated and item not in self.inductive and item not in temp_L2 and item!=link_datum[0] and str(self.re_id[item]) not in candidate]
                        candidate.extend(sample_ids)
                    candidate = candidate[:candidate_num]
                    candidate.append(str(self.re_id[link_datum[2]]))
                    random.shuffle(candidate)
                    post=' \n Pick the most suitable node from the following list: {}.'.format(', '.join(candidate))
                    #
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],self.re_id[link_datum[0]],self.re_id[link_datum[1]])+post
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])
                        middle_list=middle_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][0]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[link_datum[0]],self.re_id[link_datum[1]])+post
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])

            elif task_template['id'] == '1-2-2-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1], self.re_id[link_datum[2]], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                            add_1=(link_datum[0],self.train_L2[link_datum[0]][idx][0]) if link_datum[0]>self.train_L2[link_datum[0]][idx][0] else (self.train_L2[link_datum[0]][idx][0],link_datum[0])
                            add_2=(self.train_L2[link_datum[0]][idx][0],self.train_L2[link_datum[0]][idx][1]) if self.train_L2[link_datum[0]][idx][0]>self.train_L2[link_datum[0]][idx][1] else (self.train_L2[link_datum[0]][idx][1],self.train_L2[link_datum[0]][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1], self.re_id[link_datum[2]], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''    
                        feature_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                            add_1=(link_datum[0],self.train_L2[link_datum[0]][idx][0]) if link_datum[0]>self.train_L2[link_datum[0]][idx][0] else (self.train_L2[link_datum[0]][idx][0],link_datum[0])
                            add_2=(self.train_L2[link_datum[0]][idx][0],self.train_L2[link_datum[0]][idx][1]) if self.train_L2[link_datum[0]][idx][0]>self.train_L2[link_datum[0]][idx][1] else (self.train_L2[link_datum[0]][idx][1],self.train_L2[link_datum[0]][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段     1-2-2-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1], self.re_id[link_datum[2]], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])

                            add_1=(link_datum[0],self.L2[link_datum[0]][idx][0]) if link_datum[0]>self.L2[link_datum[0]][idx][0] else (self.L2[link_datum[0]][idx][0],link_datum[0])
                            add_2=(self.L2[link_datum[0]][idx][0],self.L2[link_datum[0]][idx][1]) if self.L2[link_datum[0]][idx][0]>self.L2[link_datum[0]][idx][1] else (self.L2[link_datum[0]][idx][1],self.L2[link_datum[0]][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1], self.re_id[link_datum[2]], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.t_L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''    
                        feature_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])

                            add_1=(link_datum[0],self.L2[link_datum[0]][idx][0]) if link_datum[0]>self.L2[link_datum[0]][idx][0] else (self.L2[link_datum[0]][idx][0],link_datum[0])
                            add_2=(self.L2[link_datum[0]][idx][0],self.L2[link_datum[0]][idx][1]) if self.L2[link_datum[0]][idx][0]>self.L2[link_datum[0]][idx][1] else (self.L2[link_datum[0]][idx][1],self.L2[link_datum[0]][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')

            elif task_template['id'] == '1-2-2-2':
                if self.mode=='train': 
                    #
                    node_list=''    
                    feature_list=''
                    if link_datum[0]>link_datum[1]:
                        f_dx_1=(link_datum[0],link_datum[1])
                    else:
                        f_dx_1=(link_datum[1],link_datum[0])
                    if link_datum[1]>link_datum[2]:
                        f_dx_2=(link_datum[1],link_datum[2])
                    else:
                        f_dx_2=(link_datum[2],link_datum[1])
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';' )
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                        add_1=(link_datum[0],self.train_L2[link_datum[0]][idx][0]) if link_datum[0]>self.train_L2[link_datum[0]][idx][0] else (self.train_L2[link_datum[0]][idx][0],link_datum[0])
                        add_2=(self.train_L2[link_datum[0]][idx][0],self.train_L2[link_datum[0]][idx][1]) if self.train_L2[link_datum[0]][idx][0]>self.train_L2[link_datum[0]][idx][1] else (self.train_L2[link_datum[0]][idx][1],self.train_L2[link_datum[0]][idx][0])
                        feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])

                elif self.mode=='val':   #1-2-2-2
                    node_list=''    
                    feature_list=''
                    if link_datum[0]>link_datum[1]:
                        f_dx_1=(link_datum[0],link_datum[1])
                    else:
                        f_dx_1=(link_datum[1],link_datum[0])
                    if link_datum[1]>link_datum[2]:
                        f_dx_2=(link_datum[1],link_datum[2])
                    else:
                        f_dx_2=(link_datum[2],link_datum[1])
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';' )
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])

                        add_1=(link_datum[0],self.L2[link_datum[0]][idx][0]) if link_datum[0]>self.L2[link_datum[0]][idx][0] else (self.L2[link_datum[0]][idx][0],link_datum[0])
                        add_2=(self.L2[link_datum[0]][idx][0],self.L2[link_datum[0]][idx][1]) if self.L2[link_datum[0]][idx][0]>self.L2[link_datum[0]][idx][1] else (self.L2[link_datum[0]][idx][1],self.L2[link_datum[0]][idx][0])
                        feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])


            elif task_template['id'] == '1-2-2-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        middle_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                            add_1=(link_datum[0],self.train_L2[link_datum[0]][idx][0]) if link_datum[0]>self.train_L2[link_datum[0]][idx][0] else (self.train_L2[link_datum[0]][idx][0],link_datum[0])
                            add_2=(self.train_L2[link_datum[0]][idx][0],self.train_L2[link_datum[0]][idx][1]) if self.train_L2[link_datum[0]][idx][0]>self.train_L2[link_datum[0]][idx][1] else (self.train_L2[link_datum[0]][idx][1],self.train_L2[link_datum[0]][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''    
                        feature_list=''
                        middle_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                            add_1=(link_datum[0],self.train_L2[link_datum[0]][idx][0]) if link_datum[0]>self.train_L2[link_datum[0]][idx][0] else (self.train_L2[link_datum[0]][idx][0],link_datum[0])
                            add_2=(self.train_L2[link_datum[0]][idx][0],self.train_L2[link_datum[0]][idx][1]) if self.train_L2[link_datum[0]][idx][0]>self.train_L2[link_datum[0]][idx][1] else (self.train_L2[link_datum[0]][idx][1],self.train_L2[link_datum[0]][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-2-2-3
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        middle_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][0]])

                            add_1=(link_datum[0],self.L2[link_datum[0]][idx][0]) if link_datum[0]>self.L2[link_datum[0]][idx][0] else (self.L2[link_datum[0]][idx][0],link_datum[0])
                            add_2=(self.L2[link_datum[0]][idx][0],self.L2[link_datum[0]][idx][1]) if self.L2[link_datum[0]][idx][0]>self.L2[link_datum[0]][idx][1] else (self.L2[link_datum[0]][idx][1],self.L2[link_datum[0]][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.t_L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''    
                        feature_list=''
                        middle_list=''
                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][0]])

                            add_1=(link_datum[0],self.L2[link_datum[0]][idx][0]) if link_datum[0]>self.L2[link_datum[0]][idx][0] else (self.L2[link_datum[0]][idx][0],link_datum[0])
                            add_2=(self.L2[link_datum[0]][idx][0],self.L2[link_datum[0]][idx][1]) if self.L2[link_datum[0]][idx][0]>self.L2[link_datum[0]][idx][1] else (self.L2[link_datum[0]][idx][1],self.L2[link_datum[0]][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]], '\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')

            elif task_template['id'] == '1-2-2-4':
                if self.mode=='train': 
                    #
                    node_list=''    
                    feature_list=''
                    middle_list=''
                    if link_datum[0]>link_datum[1]:
                        f_dx_1=(link_datum[0],link_datum[1])
                    else:
                        f_dx_1=(link_datum[1],link_datum[0])
                    if link_datum[1]>link_datum[2]:
                        f_dx_2=(link_datum[1],link_datum[2])
                    else:
                        f_dx_2=(link_datum[2],link_datum[1])
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[link_datum[0]],self.re_id[link_datum[1]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';' )
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                        middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                        add_1=(link_datum[0],self.train_L2[link_datum[0]][idx][0]) if link_datum[0]>self.train_L2[link_datum[0]][idx][0] else (self.train_L2[link_datum[0]][idx][0],link_datum[0])
                        add_2=(self.train_L2[link_datum[0]][idx][0],self.train_L2[link_datum[0]][idx][1]) if self.train_L2[link_datum[0]][idx][0]>self.train_L2[link_datum[0]][idx][1] else (self.train_L2[link_datum[0]][idx][1],self.train_L2[link_datum[0]][idx][0])
                        feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[link_datum[0]],self.re_id[link_datum[1]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])

                elif self.mode=='val':   #1-2-2-4
                    node_list=''    
                    feature_list=''
                    middle_list=''
                    if link_datum[0]>link_datum[1]:
                        f_dx_1=(link_datum[0],link_datum[1])
                    else:
                        f_dx_1=(link_datum[1],link_datum[0])
                    if link_datum[1]>link_datum[2]:
                        f_dx_2=(link_datum[1],link_datum[2])
                    else:
                        f_dx_2=(link_datum[2],link_datum[1])
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[link_datum[0]],self.re_id[link_datum[1]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';' )
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][1]])
                        middle_list=middle_list+'{}, '.format(self.re_id[self.L2[link_datum[0]][idx][0]])

                        add_1=(link_datum[0],self.L2[link_datum[0]][idx][0]) if link_datum[0]>self.L2[link_datum[0]][idx][0] else (self.L2[link_datum[0]][idx][0],link_datum[0])
                        add_2=(self.L2[link_datum[0]][idx][0],self.L2[link_datum[0]][idx][1]) if self.L2[link_datum[0]][idx][0]>self.L2[link_datum[0]][idx][1] else (self.L2[link_datum[0]][idx][1],self.L2[link_datum[0]][idx][0])
                        feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[link_datum[0]],self.re_id[link_datum[1]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\';')
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])


            elif task_template['id'] == '1-1-3-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[3]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[3]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    temp_L3.append(el)

                        node_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-1-3-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        L3=[]   #这个用作node采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[3]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[3]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        L3=[]   #这个用作node采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.t_L2[link_datum[0]]:         
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    temp_L3.append(el)

                        node_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')

            elif task_template['id'] == '1-1-3-2':
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[link_datum[0]]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[3]])


                elif self.mode=='val':   #这是测试阶段  1-1-3-2   答案不唯一
                    L3=[]   #这个用作node采样
                    for ele in self.L2[link_datum[0]]:
                        for el in self.L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                L3.append(ele+[el])
                    temp_L3=[]    #这个用作negative采样
                    for ele in self.t_L2[link_datum[0]]:         
                        for el in self.t_L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                temp_L3.append(el)
                    #
                    candidate = []
                    candidate_num = 9
                    while len(candidate) < candidate_num:
                        sample_ids = np.random.choice(list(range(1,self.item_l+self.user_l+1)), candidate_num, replace=False)
                        sample_ids = [str(self.re_id[item]) for item in sample_ids if item not in self.isolated and item not in self.inductive and item not in temp_L3 and item!=link_datum[0] and str(self.re_id[item]) not in candidate]
                        candidate.extend(sample_ids)
                    candidate = candidate[:candidate_num]
                    candidate.append(str(self.re_id[link_datum[3]]))
                    random.shuffle(candidate)
                    post=' \n Pick the most suitable node from the following list: {}.'.format(', '.join(candidate))
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])+post
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])+post
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[3]])

            elif task_template['id'] == '1-1-3-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    temp_L3.append(el)
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段      1-1-3-3
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        L3=[]   #这个用作node采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[L3[idx][0]],self.re_id[L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        L3=[]   #这个用作node采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.t_L2[link_datum[0]]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    temp_L3.append(el)
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[L3[idx][0]],self.re_id[L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '1-1-3-4': 
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[link_datum[0]]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[3]])

                elif self.mode=='val':   #这是测试阶段   1-1-3-4
                    L3=[]   #这个用作node采样
                    for ele in self.L2[link_datum[0]]:
                        for el in self.L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                L3.append(ele+[el])
                    temp_L3=[]    #这个用作negative采样
                    for ele in self.t_L2[link_datum[0]]:         
                        for el in self.t_L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                temp_L3.append(el)
                    #
                    candidate = []
                    candidate_num = 9
                    while len(candidate) < candidate_num:
                        sample_ids = np.random.choice(list(range(1,self.item_l+self.user_l+1)), candidate_num, replace=False)
                        sample_ids = [str(self.re_id[item]) for item in sample_ids if item not in self.isolated and item not in self.inductive and item not in temp_L3 and item!=link_datum[0] and str(self.re_id[item]) not in candidate]
                        candidate.extend(sample_ids)
                    candidate = candidate[:candidate_num]
                    candidate.append(str(self.re_id[link_datum[3]]))
                    random.shuffle(candidate)
                    post=' \n Pick the most suitable node from the following list: {}.'.format(', '.join(candidate))
                    #
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')+post
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])
                        middle_list=middle_list+'({},{}), '.format(self.re_id[L3[idx][0]],self.re_id[L3[idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')+post
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[3]])

            elif task_template['id'] == '1-2-3-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        feature_list=''

                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        if link_datum[2]>link_datum[3]:
                            f_dx_3=(link_datum[2],link_datum[3])
                        else:
                            f_dx_3=(link_datum[3],link_datum[2])

                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            add_1=(link_datum[0],train_L3[idx][0]) if link_datum[0]>train_L3[idx][0] else (train_L3[idx][0],link_datum[0])
                            add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                            add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    temp_L3.append(el)

                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        if link_datum[2]>link_datum[3]:
                            f_dx_3=(link_datum[2],link_datum[3])
                        else:
                            f_dx_3=(link_datum[3],link_datum[2])

                        node_list=''    
                        feature_list=''
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            add_1=(link_datum[0],train_L3[idx][0]) if link_datum[0]>train_L3[idx][0] else (train_L3[idx][0],link_datum[0])
                            add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                            add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段    1-2-3-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        L3=[]   #这个用作node采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        feature_list=''

                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        if link_datum[2]>link_datum[3]:
                            f_dx_3=(link_datum[2],link_datum[3])
                        else:
                            f_dx_3=(link_datum[3],link_datum[2])

                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])

                            add_1=(link_datum[0],L3[idx][0]) if link_datum[0]>L3[idx][0] else (L3[idx][0],link_datum[0])
                            add_2=(L3[idx][0],L3[idx][1]) if L3[idx][0]>L3[idx][1] else (L3[idx][1],L3[idx][0])
                            add_3=(L3[idx][1],L3[idx][2]) if L3[idx][1]>L3[idx][2] else (L3[idx][2],L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        L3=[]   #这个用作node采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.t_L2[link_datum[0]]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    temp_L3.append(el)

                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        if link_datum[2]>link_datum[3]:
                            f_dx_3=(link_datum[2],link_datum[3])
                        else:
                            f_dx_3=(link_datum[3],link_datum[2])

                        node_list=''    
                        feature_list=''
                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])

                            add_1=(link_datum[0],L3[idx][0]) if link_datum[0]>L3[idx][0] else (L3[idx][0],link_datum[0])
                            add_2=(L3[idx][0],L3[idx][1]) if L3[idx][0]>L3[idx][1] else (L3[idx][1],L3[idx][0])
                            add_3=(L3[idx][1],L3[idx][2]) if L3[idx][1]>L3[idx][2] else (L3[idx][2],L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')

            elif task_template['id'] == '1-2-3-2':
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[link_datum[0]]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])
                    node_list=''    #这个在tokenizer看来不占位，非常好
                    feature_list=''

                    if link_datum[0]>link_datum[1]:
                        f_dx_1=(link_datum[0],link_datum[1])
                    else:
                        f_dx_1=(link_datum[1],link_datum[0])
                    if link_datum[1]>link_datum[2]:
                        f_dx_2=(link_datum[1],link_datum[2])
                    else:
                        f_dx_2=(link_datum[2],link_datum[1])
                    if link_datum[2]>link_datum[3]:
                        f_dx_3=(link_datum[2],link_datum[3])
                    else:
                        f_dx_3=(link_datum[3],link_datum[2])

                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        add_1=(link_datum[0],train_L3[idx][0]) if link_datum[0]>train_L3[idx][0] else (train_L3[idx][0],link_datum[0])
                        add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                        add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                        feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                        count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                    target_text = task_template['target'].format(self.re_id[link_datum[3]])
                elif self.mode=='val':   #这是测试阶段   1-2-3-2
                    L3=[]   #这个用作node采样
                    for ele in self.L2[link_datum[0]]:
                        for el in self.L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                L3.append(ele+[el])
                    node_list=''    #这个在tokenizer看来不占位，非常好
                    feature_list=''

                    if link_datum[0]>link_datum[1]:
                        f_dx_1=(link_datum[0],link_datum[1])
                    else:
                        f_dx_1=(link_datum[1],link_datum[0])
                    if link_datum[1]>link_datum[2]:
                        f_dx_2=(link_datum[1],link_datum[2])
                    else:
                        f_dx_2=(link_datum[2],link_datum[1])
                    if link_datum[2]>link_datum[3]:
                        f_dx_3=(link_datum[2],link_datum[3])
                    else:
                        f_dx_3=(link_datum[3],link_datum[2])

                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1], self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(L3):  #count永远无法大于，最多等于
                        temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])

                        add_1=(link_datum[0],L3[idx][0]) if link_datum[0]>L3[idx][0] else (L3[idx][0],link_datum[0])
                        add_2=(L3[idx][0],L3[idx][1]) if L3[idx][0]>L3[idx][1] else (L3[idx][1],L3[idx][0])
                        add_3=(L3[idx][1],L3[idx][2]) if L3[idx][1]>L3[idx][2] else (L3[idx][2],L3[idx][1])
                        feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], feature_list[:-1],self.re_id[link_datum[0]],'\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                        count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                    target_text = task_template['target'].format(self.re_id[link_datum[3]])
            elif task_template['id'] == '1-2-3-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        feature_list=''
                        middle_list=''

                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        if link_datum[2]>link_datum[3]:
                            f_dx_3=(link_datum[2],link_datum[3])
                        else:
                            f_dx_3=(link_datum[3],link_datum[2])

                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            add_1=(link_datum[0],train_L3[idx][0]) if link_datum[0]>train_L3[idx][0] else (train_L3[idx][0],link_datum[0])
                            add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                            add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    temp_L3.append(el)

                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        if link_datum[2]>link_datum[3]:
                            f_dx_3=(link_datum[2],link_datum[3])
                        else:
                            f_dx_3=(link_datum[3],link_datum[2])

                        node_list=''    
                        feature_list=''
                        middle_list=''

                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            add_1=(link_datum[0],train_L3[idx][0]) if link_datum[0]>train_L3[idx][0] else (train_L3[idx][0],link_datum[0])
                            add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                            add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-2-3-3
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        L3=[]   #这个用作node采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        feature_list=''
                        middle_list=''

                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        if link_datum[2]>link_datum[3]:
                            f_dx_3=(link_datum[2],link_datum[3])
                        else:
                            f_dx_3=(link_datum[3],link_datum[2])

                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[L3[idx][0]],self.re_id[L3[idx][1]])

                            add_1=(link_datum[0],L3[idx][0]) if link_datum[0]>L3[idx][0] else (L3[idx][0],link_datum[0])
                            add_2=(L3[idx][0],L3[idx][1]) if L3[idx][0]>L3[idx][1] else (L3[idx][1],L3[idx][0])
                            add_3=(L3[idx][1],L3[idx][2]) if L3[idx][1]>L3[idx][2] else (L3[idx][2],L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        L3=[]   #这个用作node采样
                        for ele in self.L2[link_datum[0]]:
                            for el in self.L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.t_L2[link_datum[0]]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=link_datum[0]:
                                    temp_L3.append(el)

                        if link_datum[0]>link_datum[1]:
                            f_dx_1=(link_datum[0],link_datum[1])
                        else:
                            f_dx_1=(link_datum[1],link_datum[0])
                        if link_datum[1]>link_datum[2]:
                            f_dx_2=(link_datum[1],link_datum[2])
                        else:
                            f_dx_2=(link_datum[2],link_datum[1])
                        if link_datum[2]>link_datum[3]:
                            f_dx_3=(link_datum[2],link_datum[3])
                        else:
                            f_dx_3=(link_datum[3],link_datum[2])

                        node_list=''    
                        feature_list=''
                        middle_list=''

                        count=0
                        negative=random.randint(1,self.item_l+self.user_l)
                        while negative in self.isolated or negative in self.inductive or negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(1,self.item_l+self.user_l)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[L3[idx][0]],self.re_id[L3[idx][1]])

                            add_1=(link_datum[0],L3[idx][0]) if link_datum[0]>L3[idx][0] else (L3[idx][0],link_datum[0])
                            add_2=(L3[idx][0],L3[idx][1]) if L3[idx][0]>L3[idx][1] else (L3[idx][1],L3[idx][0])
                            add_3=(L3[idx][1],L3[idx][2]) if L3[idx][1]>L3[idx][2] else (L3[idx][2],L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')

            elif task_template['id'] == '1-2-3-4':
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[link_datum[0]]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                train_L3.append(ele+[el])
                    node_list=''    #这个在tokenizer看来不占位，非常好
                    feature_list=''
                    middle_list=''

                    if link_datum[0]>link_datum[1]:
                        f_dx_1=(link_datum[0],link_datum[1])
                    else:
                        f_dx_1=(link_datum[1],link_datum[0])
                    if link_datum[1]>link_datum[2]:
                        f_dx_2=(link_datum[1],link_datum[2])
                    else:
                        f_dx_2=(link_datum[2],link_datum[1])
                    if link_datum[2]>link_datum[3]:
                        f_dx_3=(link_datum[2],link_datum[3])
                    else:
                        f_dx_3=(link_datum[3],link_datum[2])

                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        add_1=(link_datum[0],train_L3[idx][0]) if link_datum[0]>train_L3[idx][0] else (train_L3[idx][0],link_datum[0])
                        add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                        add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                        feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                        count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                    target_text = task_template['target'].format(self.re_id[link_datum[3]])
                elif self.mode=='val':   #这是测试阶段  1-2-3-4
                    L3=[]   #这个用作node采样
                    for ele in self.L2[link_datum[0]]:
                        for el in self.L1[ele[1]]:
                            if el!=ele[0] and el!=link_datum[0]:
                                L3.append(ele+[el])
                    node_list=''    #这个在tokenizer看来不占位，非常好
                    feature_list=''
                    middle_list=''

                    if link_datum[0]>link_datum[1]:
                        f_dx_1=(link_datum[0],link_datum[1])
                    else:
                        f_dx_1=(link_datum[1],link_datum[0])
                    if link_datum[1]>link_datum[2]:
                        f_dx_2=(link_datum[1],link_datum[2])
                    else:
                        f_dx_2=(link_datum[2],link_datum[1])
                    if link_datum[2]>link_datum[3]:
                        f_dx_3=(link_datum[2],link_datum[3])
                    else:
                        f_dx_3=(link_datum[3],link_datum[2])

                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(L3):  #count永远无法大于，最多等于
                        temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[L3[idx][2]])
                        middle_list=middle_list+'({},{}), '.format(self.re_id[L3[idx][0]],self.re_id[L3[idx][1]])

                        add_1=(link_datum[0],L3[idx][0]) if link_datum[0]>L3[idx][0] else (L3[idx][0],link_datum[0])
                        add_2=(L3[idx][0],L3[idx][1]) if L3[idx][0]>L3[idx][1] else (L3[idx][1],L3[idx][0])
                        add_3=(L3[idx][1],L3[idx][2]) if L3[idx][1]>L3[idx][2] else (L3[idx][2],L3[idx][1])
                        feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')','\''+self.edge_feature[f_dx_1]+'\' and \''+self.edge_feature[f_dx_2]+'\' and \''+self.edge_feature[f_dx_3]+'\';')
                            #

                        count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                    target_text = task_template['target'].format(self.re_id[link_datum[3]])
            else:
                raise NotImplementedError
        



        elif task_name == 'classification':
            if self.mode=='train':   
                point=self.classification[datum_idx]
            elif self.mode=='val':
                if cate=='inductive':
                    point=self.inductive[datum_idx]   #实际上inductive这里这个point根本不能用
                elif cate=='transductive':
                    point=self.transductive[datum_idx]

            #统一进行label映射
            label=self.label_map[point]
            negative=str(np.random.choice(list(set(['fragrance','makeup','bath & body','tools & accessories','skin care','hair care']).difference({label})),1,replace=False)[0])

            if task_template['id'] == '2-1-1-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:     #对分类类别进行负label采样
                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')

                elif self.mode=='val':   #这是测试阶段  #2-1-1-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while self.t_L1[point][idx] in self.isolated and count < len(self.t_L1[point]):
                                select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count<len(self.t_L1[point]):
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count==len(self.t_L1[point]):
                                if self.t_L1[point][idx] in self.isolated:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:     #对分类类别进行负label采样
                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while self.t_L1[point][idx] in self.isolated and count < len(self.t_L1[point]):
                                select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count<len(self.t_L1[point]):
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count==len(self.t_L1[point]):
                                if self.t_L1[point][idx] in self.isolated:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')


            elif task_template['id'] == '2-1-1-2':
                if self.mode=='train': 
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   #2-1-1-2
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.t_L1[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while self.t_L1[point][idx] in self.isolated and count < len(self.t_L1[point]):
                            select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(self.t_L1[point]):
                            node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #


                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(self.t_L1[point]):
                            if self.t_L1[point][idx] in self.isolated:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)
            elif task_template['id'] == '2-2-1-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])
                            add=(point,self.train_L1[point][idx]) if point>self.train_L1[point][idx] else (self.train_L1[point][idx],point)
                            feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],feature_list[:-1], self.re_id[point], label)
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        node_list=''    
                        feature_list=''
                        count=0
                

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],feature_list[:-1], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])
                            add=(point,self.train_L1[point][idx]) if point>self.train_L1[point][idx] else (self.train_L1[point][idx],point)
                            feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],feature_list[:-1], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
                elif self.mode=='val': #2-2-1-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while self.t_L1[point][idx] in self.isolated and count < len(self.t_L1[point]):
                                select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L1[point]):

                                node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])
                                add=(point,self.t_L1[point][idx]) if point>self.t_L1[point][idx] else (self.t_L1[point][idx],point)
                                feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)
                            # 
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L1[point]):
                                if self.t_L1[point][idx] in self.isolated:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])
                                    add=(point,self.t_L1[point][idx]) if point>self.t_L1[point][idx] else (self.t_L1[point][idx],point)
                                    feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)
                            # 
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        node_list=''    
                        feature_list=''
                        count=0
                

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while self.t_L1[point][idx] in self.isolated and count < len(self.t_L1[point]):
                                select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L1[point]):

                                node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])
                                add=(point,self.t_L1[point][idx]) if point>self.t_L1[point][idx] else (self.t_L1[point][idx],point)
                                feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            # 
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L1[point]):
                                if self.t_L1[point][idx] in self.isolated:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])
                                    add=(point,self.t_L1[point][idx]) if point>self.t_L1[point][idx] else (self.t_L1[point][idx],point)
                                    feature_list=feature_list+'\'{}\'; '.format(self.edge_feature[add])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            # 
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-2-1-2':
                if self.mode=='train': 
                    #
                    node_list=''    
                    feature_list=''
                    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])
                        add=(point,self.train_L1[point][idx]) if point>self.train_L1[point][idx] else (self.train_L1[point][idx],point)
                        feature_list=feature_list + '\'{}\'; '.format(self.edge_feature[add])   #

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1],self.re_id[point])
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': #2-2-1-2
                    node_list=''    
                    feature_list=''
                    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.t_L1[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while self.t_L1[point][idx] in self.isolated and count < len(self.t_L1[point]):
                            select=list(set(list(range(len(self.t_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(self.t_L1[point]):

                            node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])
                            add=(point,self.t_L1[point][idx]) if point>self.t_L1[point][idx] else (self.t_L1[point][idx],point)
                            feature_list=feature_list + '\'{}\'; '.format(self.edge_feature[add])   #

                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(self.t_L1[point]):
                            if self.t_L1[point][idx] in self.isolated:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L1[point][idx]])
                                add=(point,self.t_L1[point][idx]) if point>self.t_L1[point][idx] else (self.t_L1[point][idx],point)
                                feature_list=feature_list + '\'{}\'; '.format(self.edge_feature[add])   #

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')

                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)
            elif task_template['id'] == '2-1-2-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        
                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val': #2-1-2-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                                select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L2[point]):

                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)
                            #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L2[point]):
                                if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)
                            #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        
                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                                select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L2[point]):

                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L2[point]):
                                if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-1-2-2':
                if self.mode=='train': 
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': #2-1-2-2     #!!!!!!!
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(self.t_L2[point]):
                            node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(self.t_L2[point]):
                            if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)
            elif task_template['id'] == '2-1-2-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val': #2-1-2-3
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                                select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L2[point]):
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node', label)
                            #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L2[point]):
                                if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                    middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node', label)
                            #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                                select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L2[point]):
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L2[point]):
                                if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                    middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-1-2-4':
                if self.mode=='train': 
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                        middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  #2-1-2-4
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(self.t_L2[point]):
                            node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #


                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(self.t_L2[point]):
                            if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)
            elif task_template['id'] == '2-2-2-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            add_1=(point,self.train_L2[point][idx][0]) if point>self.train_L2[point][idx][0] else (self.train_L2[point][idx][0],point)
                            add_2=(self.train_L2[point][idx][0],self.train_L2[point][idx][1]) if self.train_L2[point][idx][0]>self.train_L2[point][idx][1] else (self.train_L2[point][idx][1],self.train_L2[point][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1], self.re_id[point], label)
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        
                        node_list=''    
                        feature_list=''
                        
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],feature_list[:-1], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            add_1=(point,self.train_L2[point][idx][0]) if point>self.train_L2[point][idx][0] else (self.train_L2[point][idx][0],point)
                            add_2=(self.train_L2[point][idx][0],self.train_L2[point][idx][1]) if self.train_L2[point][idx][0]>self.train_L2[point][idx][1] else (self.train_L2[point][idx][1],self.train_L2[point][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],feature_list[:-1], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
                elif self.mode=='val': #2-2-2-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                                select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L2[point]):

                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)
                            
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L2[point]):
                                if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                    add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                    add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                    feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)
                            
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        
                        node_list=''    
                        feature_list=''
                        
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                                select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L2[point]):

                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            
                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L2[point]):
                                if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                    add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                    add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                    feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-2-2-2':
                if self.mode=='train': 
                    #
                    node_list=''    
                    feature_list=''
                    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                        add_1=(point,self.train_L2[point][idx][0]) if point>self.train_L2[point][idx][0] else (self.train_L2[point][idx][0],point)
                        add_2=(self.train_L2[point][idx][0],self.train_L2[point][idx][1]) if self.train_L2[point][idx][0]>self.train_L2[point][idx][1] else (self.train_L2[point][idx][1],self.train_L2[point][idx][0])
                        feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1],self.re_id[point])
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   #2-2-2-2
                    node_list=''    
                    feature_list=''
                    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(self.t_L2[point]):

                            node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                            add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                            add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                            #

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(self.t_L2[point]):
                            if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])

                                add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                            #

                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)
            elif task_template['id'] == '2-2-2-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        middle_list=''
                        
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)

                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            add_1=(point,self.train_L2[point][idx][0]) if point>self.train_L2[point][idx][0] else (self.train_L2[point][idx][0],point)
                            add_2=(self.train_L2[point][idx][0],self.train_L2[point][idx][1]) if self.train_L2[point][idx][0]>self.train_L2[point][idx][1] else (self.train_L2[point][idx][1],self.train_L2[point][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[point], label)
                            #

                            count+=1   
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        
                        node_list=''    
                        feature_list=''
                        middle_list=''
                        
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[point],negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            add_1=(point,self.train_L2[point][idx][0]) if point>self.train_L2[point][idx][0] else (self.train_L2[point][idx][0],point)
                            add_2=(self.train_L2[point][idx][0],self.train_L2[point][idx][1]) if self.train_L2[point][idx][0]>self.train_L2[point][idx][1] else (self.train_L2[point][idx][1],self.train_L2[point][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[point],negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   2-2-2-3
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    
                        feature_list=''
                        middle_list=''
                        
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                                select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L2[point]):
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L2[point]):
                                if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                    middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                    add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                    add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                    feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', label)

                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('yes')



                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露, 这个原则得坚持
                        
                        node_list=''    
                        feature_list=''
                        middle_list=''
                        
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node',negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                                select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(self.t_L2[point]):
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(self.t_L2[point]):
                                if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                    middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                    add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                    add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                    feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-2-2-4':
                if self.mode=='train': 
                    #
                    node_list=''    
                    feature_list=''
                    middle_list=''
                    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)

                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                        middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                        add_1=(point,self.train_L2[point][idx][0]) if point>self.train_L2[point][idx][0] else (self.train_L2[point][idx][0],point)
                        add_2=(self.train_L2[point][idx][0],self.train_L2[point][idx][1]) if self.train_L2[point][idx][0]>self.train_L2[point][idx][1] else (self.train_L2[point][idx][1],self.train_L2[point][idx][0])
                        feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[point])
                            #

                        count+=1   
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   #2-2-2-4
                    node_list=''    
                    feature_list=''
                    middle_list=''
                    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.t_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(self.t_L2[point]):
                            select=list(set(list(range(len(self.t_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(self.t_L2[point]):

                            node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                            add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                            add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                            feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(self.t_L2[point]):
                            if len(set(self.t_L2[point][idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[self.t_L2[point][idx][1]])
                                middle_list=middle_list+'{}, '.format(self.re_id[self.t_L2[point][idx][0]])

                                add_1=(point,self.t_L2[point][idx][0]) if point>self.t_L2[point][idx][0] else (self.t_L2[point][idx][0],point)
                                add_2=(self.t_L2[point][idx][0],self.t_L2[point][idx][1]) if self.t_L2[point][idx][0]>self.t_L2[point][idx][1] else (self.t_L2[point][idx][1],self.t_L2[point][idx][0])
                                feature_list=feature_list+'\'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2])


                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

            elif task_template['id'] == '2-1-3-1':    #3阶还是有点特殊的
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                        

                        node_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   2-1-3-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        t_L3=[]   #这个用作node采样
                        for ele in self.t_L2[point]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    t_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                                select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(t_L3):

                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)
                            #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(t_L3):
                                if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', label)
                            #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        t_L3=[]   #这个用作node采样
                        for ele in self.t_L2[point]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    t_L3.append(ele+[el])
                        

                        node_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                                select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(t_L3):

                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(t_L3):
                                if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-1-3-2':
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[point]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)


                elif self.mode=='val':  #2-1-3-2
                    t_L3=[]   #这个用作node采样
                    for ele in self.t_L2[point]:
                        for el in self.t_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                t_L3.append(ele+[el])
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(t_L3):
                            node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(t_L3):
                            if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

            elif task_template['id'] == '2-1-3-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],self.re_id[point],label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],self.re_id[point],label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                      
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point],negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段      2-1-3-3
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        t_L3=[]   #这个用作node采样
                        for ele in self.t_L2[point]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    t_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node',label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                                select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(t_L3):
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node',label)
                            #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(t_L3):
                                if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                    middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node',label)
                            #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        t_L3=[]   #这个用作node采样
                        for ele in self.t_L2[point]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    t_L3.append(ele+[el])
                      
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node',negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                                select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(t_L3):
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node',negative)
                            #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(t_L3):
                                if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                    middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node',negative)
                            #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-1-3-4':
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[point]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   #这是测试阶段   2-1-3-4
                    t_L3=[]   #这个用作node采样
                    for ele in self.t_L2[point]:
                        for el in self.t_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                t_L3.append(ele+[el])
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2],self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(t_L3):
                            node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(t_L3):
                            if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)
            elif task_template['id'] == '2-2-3-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        feature_list=''

                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1],self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            add_1=(point,train_L3[idx][0]) if point>train_L3[idx][0] else (train_L3[idx][0],point)
                            add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                            add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1],self.re_id[point], label)
                            

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])

                        node_list=''    
                        feature_list=''
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],feature_list[:-1], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            add_1=(point,train_L3[idx][0]) if point>train_L3[idx][0] else (train_L3[idx][0],point)
                            add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                            add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],feature_list[:-1], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段    2-2-3-1
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        t_L3=[]   #这个用作node采样
                        for ele in self.t_L2[point]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    t_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        feature_list=''

                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                                select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(t_L3):
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', label)
                            

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(t_L3):
                                if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                    add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                    add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                    add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                    feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', label)
                            
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        t_L3=[]   #这个用作node采样
                        for ele in self.t_L2[point]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    t_L3.append(ele+[el])

                        node_list=''    
                        feature_list=''
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node', negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                                select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(t_L3):
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(t_L3):
                                if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                    add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                    add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                    add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                    feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', negative)
                            
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-2-3-2':
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[point]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])
                    node_list=''    #这个在tokenizer看来不占位，非常好
                    feature_list=''

                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1], self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        add_1=(point,train_L3[idx][0]) if point>train_L3[idx][0] else (train_L3[idx][0],point)
                        add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                        add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                        feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], feature_list[:-1],self.re_id[point])
                            #

                        count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                    target_text = task_template['target'].format(label)
                elif self.mode=='val':   #这是测试阶段   2-2-3-2
                    t_L3=[]   #这个用作node采样
                    for ele in self.t_L2[point]:
                        for el in self.t_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                t_L3.append(ele+[el])
                    node_list=''    #这个在tokenizer看来不占位，非常好
                    feature_list=''

                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(t_L3):
                            node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                            add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                            add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                            add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                            #

                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(t_L3):
                            if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])

                                add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                    target_text = task_template['target'].format(label)
            elif task_template['id'] == '2-2-3-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        feature_list=''
                        middle_list=''

                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[point],label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            add_1=(point,train_L3[idx][0]) if point>train_L3[idx][0] else (train_L3[idx][0],point)
                            add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                            add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])

                        node_list=''    
                        feature_list=''
                        middle_list=''

                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[point],negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            add_1=(point,train_L3[idx][0]) if point>train_L3[idx][0] else (train_L3[idx][0],point)
                            add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                            add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[point],negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   2-2-3-3
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        t_L3=[]   #这个用作node采样
                        for ele in self.t_L2[point]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    t_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        feature_list=''
                        middle_list=''

                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node',label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                                select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(t_L3):
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', label)
                               #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(t_L3):
                                if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                    middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                    add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                    add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                    add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                    feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', label)
                               #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        t_L3=[]   #这个用作node采样
                        for ele in self.t_L2[point]:
                            for el in self.t_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    t_L3.append(ele+[el])

                        node_list=''    
                        feature_list=''
                        middle_list=''

                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2],feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node',negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                            while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                                select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                                idx=int(np.random.choice(select,1,replace=False)[0])
                                already_idx.append(idx)
                                count+=1
                            if count < len(t_L3):
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', negative)
                               #

                                go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                            elif count == len(t_L3):
                                if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                    source_text=temp_text
                                else:
                                    node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                    middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                    add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                    add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                    add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                    feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2], middle_list[:-2], feature_list[:-1],self.re_id[point] if cate=='transductive' else 'this new node', negative)
                               #
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
            elif task_template['id'] == '2-2-3-4':
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[point]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])
                    node_list=''    #这个在tokenizer看来不占位，非常好
                    feature_list=''
                    middle_list=''

                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        add_1=(point,train_L3[idx][0]) if point>train_L3[idx][0] else (train_L3[idx][0],point)
                        add_2=(train_L3[idx][0],train_L3[idx][1]) if train_L3[idx][0]>train_L3[idx][1] else (train_L3[idx][1],train_L3[idx][0])
                        add_3=(train_L3[idx][1],train_L3[idx][2]) if train_L3[idx][1]>train_L3[idx][2] else (train_L3[idx][2],train_L3[idx][1])
                        feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[point])
                            #

                        count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                    target_text = task_template['target'].format(label)         
                elif self.mode=='val':   #这是测试阶段  2-2-3-4
                    t_L3=[]   #这个用作node采样
                    for ele in self.t_L2[point]:
                        for el in self.t_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                t_L3.append(ele+[el])
                    node_list=''    #这个在tokenizer看来不占位，非常好
                    feature_list=''
                    middle_list=''

                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(t_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        count+=1
                        while len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0 and count < len(t_L3):
                            select=list(set(list(range(len(t_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            count+=1
                        if count < len(t_L3):
                            node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                            add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                            add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                            add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                            feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                            source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        elif count == len(t_L3):
                            if len(set(t_L3[idx]).intersection(set(self.isolated+self.inductive)))!=0:
                                source_text=temp_text
                            else:
                                node_list=node_list+'{}, '.format(self.re_id[t_L3[idx][2]])
                                middle_list=middle_list+'({},{}), '.format(self.re_id[t_L3[idx][0]],self.re_id[t_L3[idx][1]])

                                add_1=(point,t_L3[idx][0]) if point>t_L3[idx][0] else (t_L3[idx][0],point)
                                add_2=(t_L3[idx][0],t_L3[idx][1]) if t_L3[idx][0]>t_L3[idx][1] else (t_L3[idx][1],t_L3[idx][0])
                                add_3=(t_L3[idx][1],t_L3[idx][2]) if t_L3[idx][1]>t_L3[idx][2] else (t_L3[idx][2],t_L3[idx][1])
                                feature_list=feature_list+'\'{}\' and \'{}\' and \'{}\'; '.format(self.edge_feature[add_1],self.edge_feature[add_2],self.edge_feature[add_3])
                            
                                source_text =self.prefix + task_template['source'].format(self.re_id[point] if cate=='transductive' else 'A new node', node_list[:-2],middle_list[:-2], feature_list[:-1], self.re_id[point] if cate=='transductive' else 'this new node')
                            #
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                    target_text = task_template['target'].format(label)         
            else:
                raise NotImplementedError
            
        elif task_name == 'intermediate':  #暂时没加
            pass
        else:
            raise NotImplementedError
            

        input_ids = self.tokenizer.encode(    #input_ids因为自动加上了结束符号所以比tokenized_text长度长1
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
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
        

class Cora_Dataset(Dataset):#
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
        self.prefix='node represents academic paper with a specific topic, link represents a citation between the two papers. '
        self.label_map=load_pickle(os.path.join('Cora','my_data','label_map.pkl'))  #1
        self.re_id=load_pickle(os.path.join('Cora','my_data','re_id.pkl'))  #2
        self.l_max=self.args.max_text_length-8
        print(self.l_max)
        self.real_feature=load_pickle(os.path.join('Cora','my_data','real_feature.pkl')) #3
        self.train_L1=load_pickle(os.path.join('Cora','my_data','L1.pkl'))  #4
        self.train_L2=load_pickle(os.path.join('Cora','my_data','L2.pkl'))  #5  我这里弄成train_L1只是因为与amazon已有代码兼容,少改动
        self.transductive=load_pickle(os.path.join('Cora','my_data','transductive.pkl'))  #6 a list
        self.classification=load_pickle(os.path.join('Cora','my_data','classification.pkl'))  #7
        
        if self.mode=='train':
            train_p=load_pickle(os.path.join('Cora','my_data','train_path.pkl'))
            train_path=list(train_p.values())
            self.train_path=train_path
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        if self.mode=='train':
            self.compute_datum_info_train()
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
                        self.total_length += len(self.train_path) * 2   #这个是以temp为中心地乘以了2
                        which=0
                        for i in range(self.total_length - curr):
                            wh=random.randint(0,1)
                            #which=random.randint(0,1)
                            #self.datum_info.append((i + curr, key, i // 2,tems[i%2],which))   #len=5
                            self.datum_info.append((i + curr, key, i // 2,tems[wh],which))
                            which=1-which
                        curr = self.total_length
                    elif '1-1-2-1' in tems:  
                        self.total_length += len(self.train_path) * 2
                        which=2


#应该学当时Amazon，把决定template放到get_item_里面，现在这样很不好


                        for i in range(self.total_length - curr):
                            #which=random.randint(2,3)
                            wh=random.randint(0,3)
                            self.datum_info.append((i + curr, key, i // 2,tems[wh],which))
                            which=5-which
                        curr = self.total_length
                    elif '1-1-3-1' in tems:  
                        self.total_length += len(self.train_path) * 3
                        which=4
                        for i in range(self.total_length - curr):
                            #which=random.randint(4,6)
                            wh=random.randint(0,3)
                            self.datum_info.append((i + curr, key, i // 3,tems[wh],which))  
                            which=int(-1.5*which**2+14.5*which-29)
                        curr = self.total_length
            elif key == 'classification':  # 以hop水平分组，2+4+4,history不用了, 后面改写pretrain.py的时候要注意！！！！！
                for tems in self.task_list[key]:
                    if '2-1-1-1' in tems:
                        self.total_length += len(self.classification) * 2
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,tems[i % 2],'transductive'))  
                        curr = self.total_length
                    elif '2-1-2-1' in tems:

                        self.total_length += len(self.classification) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'transductive'))
                        curr = self.total_length
                    elif '2-1-3-1' in tems:

                        self.total_length += len(self.classification) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'transductive'))
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
                    if '2-1-1-1' in tems:
                        self.total_length += len(self.transductive) * 2
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 2,tems[i % 2],'transductive'))  
                        curr = self.total_length
                    elif '2-1-2-1' in tems:

                        self.total_length += len(self.transductive) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'transductive'))
                        curr = self.total_length
                    elif '2-1-3-1' in tems:

                        self.total_length += len(self.transductive) * 4
                        for i in range(self.total_length - curr):
                            self.datum_info.append((i + curr, key, i // 4,tems[i % 4],'transductive'))
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
                task_template = self.all_tasks[task_name][datum_info_idx[3]]
                if task_name=='classification':
                    cate=datum_info_idx[4]
                else:
                    which_idx=datum_info_idx[4]
                    if task_template['id']=='1-1-1-1' or task_template['id']=='1-1-1-2':
                        flip=0
                    else:
                        flip=random.randint(0,1)
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




        if task_name == 'link':
            if self.mode=='train': 
                if flip==0:
                    link_datum = self.train_path[datum_idx][which_idx]
                else:
                    link_datum = self.train_path[datum_idx][which_idx][::-1]
            elif self.mode=='val':
                pass

            if task_template['id'] == '1-1-1-1':    #记得都要加self.prefix,  记得做re_id: self.re_id
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[1]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[1]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        node_list=''    
                        count=0

                        negative=random.randint(0,2707)
                        while negative in self.train_L1[link_datum[0]] or negative==link_datum[0]:
                            negative=random.randint(0,2707)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段  #1-1-1-1
                    pass

            elif task_template['id'] == '1-1-1-2':   #记得都要加self.prefix,  记得做re_id: self.re_id
                if self.mode=='train': 
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L1[link_datum[0]][idx]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[1]])

                elif self.mode=='val':   #这是测试阶段   1-1-1-2, 这里做测试要负采样，改source_text,但不要显式强调单一正确性
                    pass

            elif task_template['id'] == '1-1-2-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[2]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[2]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露   
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.train_L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''    
                        count=0
                        negative=random.randint(0,2707)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,2707)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段    #1-1-2-1
                    pass



            elif task_template['id'] == '1-1-2-2':
                if self.mode=='train': 
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])

                elif self.mode=='val':   #这是测试阶段  1-1-2-2
                    pass


            elif task_template['id'] == '1-1-2-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[2]], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        temp_L2=[]   #这个操作保持顺序
                        for ele in self.train_L2[link_datum[0]]:
                            temp_L2.append(ele[1])
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,2707)
                        while negative in temp_L2 or negative==link_datum[0]:
                            negative=random.randint(0,2707)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-1-2-3
                    pass

            elif task_template['id'] == '1-1-2-4':
                if self.mode=='train': 
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[link_datum[0]]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[link_datum[0]])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][1]])
                        middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[link_datum[0]][idx][0]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[link_datum[0]],self.re_id[link_datum[1]])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[2]])

                elif self.mode=='val':   #这是测试阶段  1-1-2-4
                    pass

            elif task_template['id'] == '1-1-3-1':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[3]], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[3]], self.re_id[link_datum[0]])
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=link_datum[0]:
                                    temp_L3.append(el)

                        node_list=''    
                        count=0
                        negative=random.randint(0,2707)                                               
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,2707)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[negative], self.re_id[link_datum[0]])
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   1-1-3-1
                    pass

            elif task_template['id'] == '1-1-3-2':
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[link_datum[0]]:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0]:
                                train_L3.append(ele+[el])
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],self.re_id[link_datum[0]])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], self.re_id[link_datum[0]])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[3]])


                elif self.mode=='val':   #这是测试阶段  1-1-3-2   答案不唯一
                    pass

            elif task_template['id'] == '1-1-3-3':
                if self.mode=='train': 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2], middle_list[:-2],self.re_id[link_datum[3]], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=link_datum[0]:
                                    train_L3.append(ele+[el])
                        temp_L3=[]    #这个用作negative采样
                        for ele in self.train_L2[link_datum[0]]:
                            for el in self.train_L1[ele[1]]:
                                if el!=link_datum[0]:
                                    temp_L3.append(el)
                        node_list=''
                        middle_list=''    
                        count=0
                        negative=random.randint(0,2707)
                        while negative in temp_L3 or negative==link_datum[0]:
                            negative=random.randint(0,2707)

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[negative], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段      1-1-3-3
                    pass
            elif task_template['id'] == '1-1-3-4': 
                if self.mode=='train': 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[link_datum[0]]:
                        for el in self.train_L1[ele[1]]:
                            if el!=link_datum[0]:
                                train_L3.append(ele+[el])
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2],self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[link_datum[0]], node_list[:-2],middle_list[:-2], self.re_id[link_datum[0]],'('+str(self.re_id[link_datum[1]])+','+str(self.re_id[link_datum[2]])+')')
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(self.re_id[link_datum[3]])

                elif self.mode=='val':   #这是测试阶段   1-1-3-4
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
            negative=str(np.random.choice(list(set(['theory','reinforcement learning','genetic algorithms','neural networks','probabilistic methods','case based','rule learning']).difference({label})),1,replace=False)[0])

            if task_template['id'] == '2-1-1-1':
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:     #对分类类别进行负label采样
                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text

                        target_text = task_template['target'].format('no')

                elif self.mode=='val':   #这是测试阶段  #2-1-1-1
                    pass


            elif task_template['id'] == '2-1-1-2':
                if self.mode!=None: 
                    #
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L1[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L1[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L1[point][idx]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   #2-1-1-2
                    pass
            
            elif task_template['id'] == '2-1-2-1':
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        
                        node_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val': #2-1-2-1
                    pass
            elif task_template['id'] == '2-1-2-2':
                if self.mode!=None: 
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val': #2-1-2-2     #!!!!!!!
                    pass
            elif task_template['id'] == '2-1-2-3':
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        node_list=''
                        middle_list=''    
                        count=0

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                            middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val': #2-1-2-3
                    pass
            elif task_template['id'] == '2-1-2-4':
                if self.mode!=None: 
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(self.train_L2[point]):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(self.train_L2[point])))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[self.train_L2[point][idx][1]])
                        middle_list=middle_list+'{}, '.format(self.re_id[self.train_L2[point][idx][0]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':  #2-1-2-4
                    pass

            elif task_template['id'] == '2-1-3-1':    #3阶还是有点特殊的
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                        

                        node_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段   2-1-3-1
                    pass
            elif task_template['id'] == '2-1-3-2':
                if self.mode!=None: 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[point]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])
                    node_list=''    
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)


                elif self.mode=='val':  #2-1-3-2
                    pass

            elif task_template['id'] == '2-1-3-3':
                if self.mode!=None: 
                    rand_prob = random.random()
                    if rand_prob > 0.5:
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                        node_list=''    #这个在tokenizer看来不占位，非常好
                        middle_list=''
                        count=0
                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],self.re_id[point],label)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   #此时的temp_text是合格的

                            #处理得到新的source_text,并且为了保险是采用完全重新赋值的方式

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2], middle_list[:-2],self.re_id[point],label)
                            #

                            count+=1   #超长本身会报错吗？
                            #不会，self.tokenizer.tokenize(...) can be used for any long sequence to detect. Good!
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text
                        #我要做到当while结束的时候source_text已经ok了
                        target_text = task_template['target'].format('yes')

                    else:  #得做负例采样, 结合isolated, inductive, L(第二层次子图)就可以采，不能用t_L,不然信息泄露
                        train_L3=[]   #这个用作node采样
                        for ele in self.train_L2[point]:
                            for el in self.train_L1[ele[1]]:
                                if el!=ele[0] and el!=point:
                                    train_L3.append(ele+[el])
                      
                        node_list=''
                        middle_list=''    
                        count=0
                        

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point],negative)
                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        already_idx=[]
                        while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                            temp_text=source_text   

                            #只要在这个意义上不重复就行了
                            select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                            idx=int(np.random.choice(select,1,replace=False)[0])
                            already_idx.append(idx)
                            node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                            middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                            source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point], negative)
                            #

                            count+=1  
                            go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                        if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                            pass
                        else:
                            source_text=temp_text


                        target_text = task_template['target'].format('no')
                elif self.mode=='val':   #这是测试阶段      2-1-3-3
                    pass
            elif task_template['id'] == '2-1-3-4':
                if self.mode!=None: 
                    train_L3=[]   #这个用作node采样
                    for ele in self.train_L2[point]:
                        for el in self.train_L1[ele[1]]:
                            if el!=ele[0] and el!=point:
                                train_L3.append(ele+[el])
                    node_list=''    
                    middle_list=''
                    count=0
                    source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2],self.re_id[point])
                    go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    already_idx=[]
                    while go_on and count < len(train_L3):  #count永远无法大于，最多等于
                        temp_text=source_text   

                            #只要在这个意义上不重复就行了
                        select=list(set(list(range(len(train_L3)))).difference(set(already_idx)))
                        idx=int(np.random.choice(select,1,replace=False)[0])
                        already_idx.append(idx)
                        node_list=node_list+'{}, '.format(self.re_id[train_L3[idx][2]])
                        middle_list=middle_list+'({},{}), '.format(self.re_id[train_L3[idx][0]],self.re_id[train_L3[idx][1]])

                        source_text =self.prefix + task_template['source'].format(self.re_id[point], node_list[:-2],middle_list[:-2], self.re_id[point])
                            #

                        count+=1  

                        go_on=True if len(self.tokenizer.tokenize(source_text))+1 < self.l_max else False
                    if len(self.tokenizer.tokenize(source_text))+1 <= self.l_max:
                        pass
                    else:
                        source_text=temp_text

                    target_text = task_template['target'].format(label)

                elif self.mode=='val':   #这是测试阶段   2-1-3-4
                    pass
            
            else:
                raise NotImplementedError
            
        elif task_name == 'intermediate':  #暂时没加
            pass
        else:
            raise NotImplementedError
            

        input_ids = self.tokenizer.encode(    #input_ids因为自动加上了结束符号所以比tokenized_text长度长1
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        #print('hereeeeeeeeeeeeeeeeeeeeee')









        
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
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



def get_loader(args, task_list, sample_numbers, split='toys', mode='train', 
               batch_size=16, workers=4, distributed=False):

    tokenizer = T5TokenizerFast.from_pretrained(args.backbone)

    if split == 'Cora' or split=='PubMed':
        from all_beauty_templates import all_tasks as task_templates
        from arxiv import Arxiv_Dataset  #if train label show?
        dataset = Arxiv_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )
    elif split=='Arxiv':
        from all_beauty_templates import all_tasks as task_templates
        from arxiv import Arxiv_Dataset
        dataset = Arxiv_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )
    else:
        from all_beauty_templates import all_tasks as task_templates

        dataset = beauty_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn, drop_last=False)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,    #这里可以单独设置的大胆一点
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        #len_val=len(dataset)
        
    return loader
