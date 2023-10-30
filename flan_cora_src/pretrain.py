import collections
from dis import dis
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
from param import parse_args
from pretrain_data import get_loader#, len_val
from utils import LossMeter
from dist_utils import reduce_dict, new_reduce_dict

_use_native_amp = False
_use_apex = False

_use_native_amp = True
from torch.cuda.amp import autocast

from trainer_base import TrainerBase


# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True,val_list=None):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)   #实际上这里传进去的train没有用

        ###assert args.whole_word_embed
        from pretrain_model import P5Pretraining
        #print()
        #print(len(self.val_loader))
        #print(len(self.val_loader.dataset))
        #print()
        #print(',,,mmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')

        model_kwargs = {}

        model_class = P5Pretraining

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config, **model_kwargs)



          #32128----->32100
        self.model.resize_token_embeddings(self.tokenizer.vocab_size+2708)   #这个我肯定要用

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:        #不从这里进，保持None
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            self.start_epoch = int(args.load.split('Epoch-')[-1])

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()

#        ckpt_path="begin_Aa_nflan_s_2_1e-05_8_Cora_mend.pth"
 #       self.load_checkpoint(ckpt_path)
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        #if self.args.train=='Cora':
         #   self.model.shared.weight.requires_grad=False

        if args.multiGPU and not args.inference:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu])
                                 #find_unused_parameters=True
                                 #)
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        self.val_list=val_list

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:       #从来不dry
            results = self.evaluate_epoch()   

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            #if 't5' in self.args.backbone:
            project_name = "P5_Pretrain"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        #global_step = 0
        for epoch in range(self.args.epoch):
            global_step=0

            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)    #keep in mind this

            # Train
            self.model.train()     # 这里设置好了

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}    #per-epoch的
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):   #step_i可用来操作acc_step
                torch.cuda.empty_cache()
            
                dist.barrier()

                if self.args.fp16 and _use_native_amp:
                    pass
                else:
                    if self.args.distributed:
                        #
                        dddd = next(self.model.parameters()).device

                        input_ids = batch['input_ids'].to(dddd)
                        lm_labels = batch["target_ids"].to(dddd)

                        loss_weights = batch["loss_weights"].to(dddd)
                        B, L = lm_labels.size()

                        output = self.model(  #这里本质也是调用forward
                            input_ids=input_ids,
                            real_feature=self.train_loader.dataset.real_feature.to(dddd),   
                            labels=lm_labels,
                            return_dict=True
                        )

                        lm_mask = lm_labels != -100
                        lm_mask = lm_mask.float()

                        loss = output['loss']

                        loss = loss.view(B, L) * lm_mask   #注意一下loss的size

                        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)   #这里之后变1维的了

                        results = {}     #Real output for our model

                        results['loss'] = (loss * loss_weights).mean()    #这个loss_weight是per_batch(per_example)的
                        results['total_loss'] = loss.detach().sum()
                        results['total_loss_count'] = len(loss)

                        task_counts = {task: 0 for task in self.model.module.losses}
                        task_loss = {task: 0 for task in self.model.module.losses}

                        for _loss, task in zip(loss.detach(), batch['task']):
                            task_loss[task] += _loss
                            task_counts[task] += 1

                        for task in self.model.module.losses:
                            if task_counts[task] > 0:
                                results[f'{task}_loss'] = task_loss[task]
                                results[f'{task}_loss_count'] = task_counts[task]

                        #results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']/self.args.gradient_accumulation_steps
                dist.barrier()
                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                #print(self.model.module.encoder.embed_tokens(torch.tensor([586]).to(batch.device)).grad)

                loss = loss.detach()

                # Update Parameters
                if step_i % self.args.gradient_accumulation_steps==0:
                    #
                    if self.args.clip_grad_norm > 0:
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                        elif self.args.fp16 and _use_apex:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()    #梯度更新

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
 
                    for param in self.model.parameters():    
                        param.grad = None

                global_step += 1
                if epoch==0 and global_step in [-100]:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"s_sp1_{}_8_{}_{}_mid_n_{}.pth".format(self.args.lr,self.args.train,self.args.weight_decay,global_step))
                if global_step==len(self.train_loader)//2:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"Aa_nflan_s_{}_{}_8_{}_mid2.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)//4:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"Aa_nflan_s_{}_{}_8_{}_mid1.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*3//4:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"Aa_nflan_s_{}_{}_8_{}_mid3.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)//8:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"Aa_nflan_s_{}_{}_8_{}_mmid1.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*3//8:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"Aa_nflan_s_{}_{}_8_{}_mmid2.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*5//8:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"Aa_nflan_s_{}_{}_8_{}_mmid3.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*7//8:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"Aa_nflan_s_{}_{}_8_{}_mend.pth".format(epoch+1,self.args.lr,self.args.train))

                dist.barrier()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    lr=self.optim.param_groups[-1]['lr']
                    #try:
                     #   lr = self.optim.get_lr()[0]
                    #except AttributeError:
                     #   lr = self.args.lr

                for k, v in results.items():    #本project 标准写法
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 1==0:       #问题是不是有可能出在这？
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'    #results是per-interation地产生的

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:   #看一下这两玩意的type，我能不能不累积，只打印该iteration的loss不就好了吗，拒绝append?
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']     #epoch_results具有累计性质
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results,average=False)    #Global信息拿到#

            dist.barrier()


            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                losses_str += '\n'
                print(losses_str)      #这个打印频率 once per epoch

            dist.barrier()

            if self.verbose:
                
                #if epoch+1==self.args.epoch:
                #torch.save(self.model.state_dict(),'{}_pretrain_link.pth'.format(epoch+1))
                torch.save(self.model.state_dict(),"Aa_nflan_s_{}_{}_8_{}_end.pth".format(epoch+1,self.args.lr,self.args.train))

            dist.barrier()

 ##           if epoch+1==self.args.epoch:     #关键代码,inference,
                #把inference和train分离试试



 ##               valid_results = self.evaluate_epoch()#.to('cuda')  
 ##               dist.barrier()

   ##             valid_results = new_reduce_dict(valid_results)   #
    ##            dist.barrier()

      ##          if self.verbose:
        ##            print()
          ##          print()
            ##        for kk in valid_results.keys():#
              ##          if kk.startswith('1'):
                ##            valid_results[kk]=valid_results[kk].item()/self.val_loader.dataset.len_link[kk]
                  ##      elif kk.endswith('inductive'):
                    ##        valid_results[kk]=valid_results[kk].item()/self.val_loader.dataset.len_inductive[kk.rstrip('-inductive')]
                      ##  elif kk.endswith('transductive'):
                        ##    valid_results[kk]=valid_results[kk].item()/self.val_loader.dataset.len_transductive[kk.rstrip('-transductive')]
                    ##print(valid_results)
                   ## print()
                    ##print()

                ##dist.barrier()

                #per-epoch地将测试结果写入一个txt文件
                ##if self.verbose:
                  ##  acc_file=open('acc.txt','a')
                    ##acc_file.write(str(epoch+1)+'\n')
                    ##acc_file.write(str(valid_results)+'\n\n')
                    ##acc_file.close()
                ##dist.barrier()
            
            

    def test(self):   #没有封装DDP
        special=[]
        for epoch in range(4*self.args.epoch):
            #load 对应模型
            #ckpt_path = "{}_{}_{}.pth".format(epoch+1,self.args.lr,self.args.world_size)  #GPU数目注意一下
            if epoch<=-100:
                spp=special[epoch]
                ckpt_path="s_sp1_{}_8_{}_{}_mid_n_{}.pth".format(self.args.lr,self.args.train,self.args.weight_decay,spp)
            elif (epoch+1)%4==1:
                #continue
                ckpt_path = "Aa_nflan_s_{}_{}_8_{}_mmid1.pth".format(epoch//4+1,self.args.lr,self.args.train) 
            elif (epoch+1)%4==2:
                
                ckpt_path = "Aa_nflan_s_{}_{}_8_{}_mmid2.pth".format(epoch//4+1,self.args.lr,self.args.train) 
            elif (epoch+1)%4==3:
                
                #continue
                ckpt_path = "Aa_nflan_s_{}_{}_8_{}_mmid3.pth".format(epoch//4+1,self.args.lr,self.args.train) 
            elif (epoch+1)%4==0:
                ckpt_path = "Aa_nflan_s_{}_{}_8_{}_mend.pth".format(epoch//4+1,self.args.lr,self.args.train)
       #     ckpt_path='L4Aa_nflan_s_{}_0.0001_8_Cora_end.pth'.format(epoch)

            self.load_checkpoint(ckpt_path)
            self.model = self.model.to(self.args.gpu)
            #
            valid_results = self.evaluate_epoch()
            dist.barrier()

            valid_results = new_reduce_dict(valid_results)   #
            dist.barrier()

            if self.verbose:
                print()
                print()
                for kk in valid_results.keys():#
                    if kk.startswith('1'):
                        valid_results[kk]=valid_results[kk].item()/self.val_loader.dataset.len_link[kk]
                    elif kk.endswith('inductive'):
                        valid_results[kk]=valid_results[kk].item()/self.val_loader.dataset.len_inductive[kk.rstrip('-inductive')]
                    elif kk.endswith('transductive'):
                        if self.args.train=='Cora' or self.args.train=='Arxiv':
                            valid_results[kk]=valid_results[kk].item()/self.val_loader.dataset.len_transductive
                        else:
                            valid_results[kk]=valid_results[kk].item()/self.val_loader.dataset.len_transductive[kk.rstrip('-transductive')]
                print(valid_results)
                print()
                print()

            dist.barrier()

                #per-epoch地将测试结果写入一个txt文件
            if self.verbose:
                acc_file=open('cora_1e-3.txt','a')                           #文件名字注意一下
                if epoch<=-100:
                    acc_file.write(str(spp)+'\n')
                elif (epoch+1)%4==1:
                    acc_file.write(str(epoch//4+1)+'_mid1'+'\n')
                elif (epoch+1)%4==2:
                    acc_file.write(str(epoch//4+1)+'_mid2'+'\n')
                elif (epoch+1)%4==3:
                    acc_file.write(str(epoch//4+1)+'_mid3'+'\n')
                else:
                    acc_file.write(str(epoch//4+1)+'_end'+'\n')
                #acc_file.write(str(epoch+1)+'\n')
                acc_file.write(str(valid_results)+'\n\n')
                acc_file.close()
            dist.barrier()


    def evaluate_epoch(self):    #关键代码#     per-template地给出acc, 其中classification的template给两acc
        #得到一个字典：key: template-id(+cate), value:正确个数
        ACC={}
        for k in list(self.val_list.keys()):
            if k=='link':
                templates=[]
                for tems in self.val_list[k]:
                    templates=templates+tems
                for thing in templates:
                    ACC[thing]=0
            elif k=='classification':
                if self.args.train=='Cora' or self.args.train=='Arxiv':
                    templates=[]
                    for tems in self.val_list[k]:
                        templates=templates+tems
                    for thing in templates:
                        ACC[thing+'-'+'transductive']=0
                else:
                    templates=[]
                    for tems in self.val_list[k]:
                        templates=templates+tems
                    for thing in templates:
                        ACC[thing+'-'+'inductive']=0
                        ACC[thing+'-'+'transductive']=0
        self.model.eval()
        with torch.no_grad():
            for step_i, batch in tqdm(enumerate(self.val_loader)):   #每张卡单独inference完毕后per_inference只做一次结果同步

                if self.args.distributed:
                    results = self.model.g_step(batch,real=self.val_loader.dataset.real_feature)
                else:
                    results = self.model.g_step(batch)   #results为一个长度为B的list,元素类型为str

                for iiid in range(len(results)):    #这里后期引入其他任务后还要改
                    task=batch['task'][iiid]
                    temp_id=batch['temp_ids'][iiid]
                    if task=='classification':
                        
                        cate=batch['cate'][iiid] #无论如何batch都会传出cate
                        if temp_id.endswith('2') or temp_id.endswith('4') or temp_id.endswith('6') or temp_id.endswith('7'):  #label, length may be greater than 1
                            if results[iiid].lower()==batch['target_text'][iiid]:
                                ACC[temp_id+'-'+cate]+=1
                        else:   #yes-no
                            focus=results[iiid].split(' ')[0]
                            if focus.lower() in ('yes', 'true', 't', 'y'):
                                ff=True
                            elif focus.lower() in ('no', 'false', 'f', 'n'):
                                ff=False
                            else:
                                ff=focus
                            fff=True if batch['target_text'][iiid]=='yes' else False
                            if ff==fff:
                                ACC[temp_id+'-'+cate]+=1

                    elif task=='link':
                        focus=results[iiid].split(' ')[0].lower()
                        if temp_id.endswith('2') or temp_id.endswith('4'):  #id
                            if focus==batch['target_text'][iiid]:
                                ACC[temp_id]+=1
                        else:     #yes-no
                            if focus.lower() in ('yes', 'true', 't', 'y'):
                                ff=True
                            elif focus.lower() in ('no', 'false', 'f', 'n'):
                                ff=False
                            else:
                                ff=focus
                            fff=True if batch['target_text'][iiid]=='yes' else False
                            if ff==fff:
                                ACC[temp_id]+=1

                dist.barrier()

            return ACC   #ACC的value需要全部变成tensor并且to device, 最后还要同步前统一to cuda


def main_worker(gpu, args):     # the gpu is the args.local_rank
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    # define the prompts used in training
    if not args.inference:
        print(f'Building train loader at GPU {gpu}')
        if args.train == 'Cora':
            train_task_list = {
        ###    'classification':[['6-6-6-6'],['2-1-1-2','2-3-1-2'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']]
     #       'link':[['1-1-1-1','1-1-1-2','1-3-1-1','1-3-1-2'],['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4','1-3-2-1','1-3-2-2','1-3-2-3','1-3-2-4'],['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4','1-3-3-1','1-3-3-2','1-3-3-3','1-3-3-4']],
            'classification':[['6-6-6-6','6-6-6-7'],['2-3-1-2','2-1-1-2'],['2-3-2-2','2-3-2-4','2-1-2-2','2-1-2-4'],['2-3-3-2','2-3-3-4','2-1-3-2','2-1-3-4']]
            }
        elif args.train=='Arxiv':
            train_task_list = {
            'classification':[['6-6-6-6'],['2-1-1-2','2-3-1-2'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']]
      #      'link':[['1-1-1-1','1-1-1-2','1-3-1-1','1-3-1-2'],['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4','1-3-2-1','1-3-2-2','1-3-2-3','1-3-2-4'],['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4','1-3-3-1','1-3-3-2','1-3-3-3','1-3-3-4']]
            #'classification':[['5-5-5-5','6-6-6-6'],['2-1-1-1','2-1-1-2','2-3-1-1','2-3-1-2'],['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4','2-3-2-1','2-3-2-2','2-3-2-3','2-3-2-4'],['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4','2-3-3-1','2-3-3-2','2-3-3-3','2-3-3-4']]
            }
        else:
            train_task_list = {
            'link':[['1-1-1-1','1-1-1-2','1-2-1-1','1-2-1-2'],['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4','1-2-2-1','1-2-2-2','1-2-2-3','1-2-2-4'],['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4','1-2-3-1','1-2-3-2','1-2-3-3','1-2-3-4']],
            'classification':[['2-1-1-1','2-1-1-2','2-2-1-1','2-2-1-2'],['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4','2-2-2-1','2-2-2-2','2-2-2-3','2-2-2-4'],['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4','2-2-3-1','2-2-3-2','2-2-3-3','2-2-3-4']]
            }

        train_sample_numbers = {}
    #train_sample_numbers在我这根本没用到
        train_loader = get_loader(
            args,
            train_task_list,
            train_sample_numbers,
            split=args.train, 
            mode='train',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
            )

        if args.gpu==0:
            print('Length of train dataset:', len(train_loader.dataset))
        trainer = Trainer(args,train_loader= train_loader,  train=True)    # Remember to set 'train' = True
        trainer.train()
    # define the prompts used in validation
    if args.inference:
        print(f'Building val loader at GPU {gpu}')
        if args.valid == 'Cora':
            val_task_list = {
        #    'classification':[['6-6-6-6'],['2-1-1-2','2-3-1-2'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']]
            'classification':[['6-6-6-6','6-6-6-7'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4'],['2-1-1-2','2-3-1-2']]
            }
        elif args.valid=='Arxiv':
            val_task_list = {
            #'classification':[['5-5-5-5','6-6-6-6'],['2-1-1-1','2-1-1-2','2-3-1-1','2-3-1-2'],['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4','2-3-2-1','2-3-2-2','2-3-2-3','2-3-2-4'],['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4','2-3-3-1','2-3-3-2','2-3-3-3','2-3-3-4']]
            'classification':[['2-3-1-2'],['6-6-6-6']]#,['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']]
            }
        else:
            val_task_list = {
            'link':[['1-1-1-1','1-1-1-2','1-2-1-1','1-2-1-2'],['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4','1-2-2-1','1-2-2-2','1-2-2-3','1-2-2-4'],['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4','1-2-3-1','1-2-3-2','1-2-3-3','1-2-3-4']],
            'classification':[['2-1-1-1','2-1-1-2','2-2-1-1','2-2-1-2'],['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4','2-2-2-1','2-2-2-2','2-2-2-3','2-2-2-4'],['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4','2-2-3-1','2-2-3-2','2-2-3-3','2-2-3-4']]
            }
        val_sample_numbers = {}
        val_loader = get_loader(
            args,
            val_task_list,
            val_sample_numbers,
            split=args.valid, 
            mode='val',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )

        if args.gpu==0:
            print('Length of test dataset:', len(val_loader.dataset))

        trainer = Trainer(args, val_loader= val_loader, train=False,val_list=val_task_list)   
        trainer.test()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'toys' in args.train:
        dsets.append('toys')
    if 'beauty' in args.train:
        dsets.append('beauty')
    if 'sports' in args.train:
        dsets.append('sports')
    if 'Cora' in args.train:
        dsets.append('Cora')
    if 'Arxiv' in args.train:
        dsets.append('Arxiv')
    comments.append(''.join(dsets))
    #if args.backbone:
    comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
