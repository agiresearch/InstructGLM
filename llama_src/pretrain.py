import collections
import pickle
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
from pretrain_data import get_loader,load_pickle #, len_val
from utils import LossMeter
from dist_utils import reduce_dict, new_reduce_dict

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

_use_native_amp = False
_use_apex = False

_use_native_amp = True
from torch.cuda.amp import autocast

from trainer_base import TrainerBase
from pretrain_model import P5Pretraining

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

class FP(nn.Module):
    def __init__(self,llama_embed,real):
        super(FP,self).__init__()
       # self.trans_3=nn.Linear(512,4096,bias=False)
#        self.trans_1=nn.Linear(128,512,bias=False)  
        self.trans_2=nn.Linear(512,4096,bias=False)
        self.trans_1=nn.Linear(768,512,bias=False)
   #     self.coff = nn.Parameter(torch.Tensor([1/6,1/6,1/6,1/6,1/6,1/6]))
        self.rac=nn.ELU()
      #  self.rac2=nn.ELU()
        self.sln=nn.LayerNorm(512)
 #       self.sln2=nn.LayerNorm(512)
     #   self.sln2=nn.LayerNorm(512)
        #初始化：
        for m in self.modules():#这也可以调
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)
#        self.coff = nn.Parameter(torch.Tensor([1/6,1/6,1/6,1/6,1/6,1/6]))
        self.embed_tokens=llama_embed

 #       self.real_1=real[:,:128]
  #      self.real_2=real[:,128:256]
   #     self.real_3=real[:,256:384]
    #    self.real_4=real[:,384:512]
     #self.real_5=real[:,512:640]
      #  self.real_6=real[:,640:]
        self.real=real
        #

    def forward(self, input_ids):
       # real_feature=self.coff[0]*self.real_1+self.coff[1]*self.real_2+self.coff[2]*self.real_3+self.coff[3]*self.real_4+self.coff[4]*self.real_5+self.coff[5]*self.real_6
        transfered=self.trans_2(self.rac(self.sln(self.trans_1(self.real)))) #+ self.trans_3(self.real_feature)
       # inputs_embeds=transfered[input_ids]

     #   self.embed_tokens.weight.data[-169343:]=torch.zeros(169343,768).to(input_ids.device)
        inputs_embeds = transfered[input_ids] + self.embed_tokens[input_ids] ### embedding step - add HERE ###
        #self.embed_tokens.weight.data[-169343:]=transfered[-169343:]

        return inputs_embeds


# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True,val_list=None):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)   #实际上这里传进去的train没有用


        model_class = P5Pretraining
        self.m_class=P5Pretraining     #测试的时候有用

        config = self.create_config()
        self.m_config=config     #测试的时候有用
        self.tokenizer = self.create_tokenizer()

        re_start=2
        if train:  #若测试,则全部放到test里初始model, per checkpoint地初始

            self.model = self.create_model(model_class, config)
            self.model.tokenizer = self.tokenizer   #这一步骤对generate必要,记得附带

            self.model = prepare_model_for_int8_training(self.model)

            #在这里搞成peft-model
            lora_r=16 
            lora_alpha=16
            lora_target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head']
            lora_dropout=0.05

            LORA_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=lora_target_modules,lora_dropout=lora_dropout,bias="none",task_type="CAUSAL_LM")
            if re_start!=2:
                self.model = get_peft_model(self.model, LORA_config)

            if self.verbose and re_start!=2:
                print()
                self.model.print_trainable_parameters()
                print()
            dist.barrier()
            #********re-start
        
            if re_start==1:
                print('Main model re-starting')
                doc_prefix='./llama_2_mid3/'
                for gg in range(32):
                    self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gosqa_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Goska_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gosva_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gosoa_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gosqb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Goskb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gosvb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gosob_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.lm_head.lora_A.default.weight.data=load_pickle(doc_prefix+"Goslm_a_{}.pkl".format(self.args.lr)).data.to(args.gpu)
                self.model.base_model.model.lm_head.lora_B.default.weight.data=load_pickle(doc_prefix+"Goslm_b_{}.pkl".format(self.args.lr)).data.to(args.gpu)
                print('Main model loaded.')
                
                if self.verbose:
                    self.model.save_pretrained("Go_restart_again")
                dist.barrier()

            if re_start==2:
                from peft import PeftModel, PeftConfig
                peft_model_id ='./Go_restart'
                print('now we are loading peft model')
           ##     config = PeftConfig.from_pretrained(peft_model_id)

                self.model = PeftModel.from_pretrained(self.model, peft_model_id)
                for n,p in self.model.named_parameters():
                    if 'lora' in n:
                        p.requires_grad_()
                if self.verbose:
                    print()
                    self.model.print_trainable_parameters()
                    print()
                
                #print(self.model.base_model.model.lm_head.lora_A.default.weight.requires_grad)
                #print(self.model.base_model.model.lm_head.lora_A.default.weight)
                #print(self.model.base_model.model.lm_head.weight.dtype)

        #        if hasattr(self.model, "enable_input_require_grads"):
                    #
       #             self.model.enable_input_require_grads()
      #          else:

     #               def make_inputs_require_grad(module, input, output):
                        #
    #                    output.requires_grad_(True)

   #                 self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                                                                                                # enable gradient checkpointing for memory efficiency
  #              self.model.gradient_checkpointing_enable()
            #*******
            self.model = self.model.to(args.gpu)
        else:
            self.model = self.create_model(model_class, config)
            self.model.tokenizer = self.tokenizer   #这一步骤对generate必要,记得附带

            for name, param in self.model.named_parameters():
        # freeze base model's layers
                param.requires_grad = False

    # cast all non INT8 parameters to fp32
            for param in self.model.parameters():
                if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                    param.data = param.data.to(torch.float32)

            #在这里搞成peft-model
            lora_r=16 
            lora_alpha=16
            lora_target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head']
            lora_dropout=0.05

            LORA_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=lora_target_modules,lora_dropout=lora_dropout,bias="none",task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, LORA_config)

            if self.verbose:
                print()
                self.model.print_trainable_parameters()
                print()
            dist.barrier()
            self.model = self.model.to(args.gpu)
           
 #           print(self.model.base_model.model.model.layers[0].self_attn.k_proj.lora_A.default.weight.dtype)



      #  self.model.resize_token_embeddings(self.tokenizer.vocab_size+1+169343) #若llama加link,则必须用

        #              # add new modules, i.e. the first_model
        if train:
            self.first_model=FP(llama_embed=self.train_loader.dataset.llama_embed.to(args.gpu),real=self.train_loader.dataset.real_feature.to(args.gpu))
        else:
            self.first_model=FP(llama_embed=self.val_loader.dataset.llama_embed.to(args.gpu),real=self.val_loader.dataset.real_feature.to(args.gpu))

        #****re-start
        re_start=3
        if train and re_start==1:
            print('All processes re-starting first-model')
            ckpt_path="Gfirst_s_1_0.0003_8_Arxiv_mid3.pth"
        
            self.load_checkpoint(ckpt_path)
            if self.verbose:
                print(self.first_model.state_dict()['sln1.weight'])
                print(self.first_model.state_dict().keys())
                
                print(self.first_model.trans_1.weight)

            print('first_model loaded.')
        if train and re_start==2:
            from utils import load_state_dict
            ckpt_path="Gfirst_s_1_0.0003_8_Arxiv_mid3.pth"
            state_dict = load_state_dict(ckpt_path, 'cpu')
            
            self.first_model.sln1.bias.data=state_dict['sln1.bias']
            self.first_model.sln1.weight.data=state_dict['sln1.weight']
            self.first_model.sln2.weight.data=state_dict['sln2.weight']
            self.first_model.sln2.bias.data=state_dict['sln2.bias']
            self.first_model.trans_1.weight.data=state_dict['trans_1.weight']
   #         print(self.first_model.trans_1.weight)
            self.first_model.trans_2.weight.data=state_dict['trans_2.weight']
            self.first_model.trans_3.weight.data=state_dict['trans_3.weight']
        if train and re_start==3:
      #      self.first_model.coff.data=load_pickle('./Gofirst_restart_again/coff.pkl')
            self.first_model.sln.bias.data=load_pickle('./Gofirst_restart/sln_bias.pkl')
            self.first_model.sln.weight.data=load_pickle('./Gofirst_restart/sln_weight.pkl')
       #     self.first_model.sln2.weight.data=load_pickle('./Gofirst_restart_again/sln2_weight.pkl')
        #    self.first_model.sln2.bias.data=load_pickle('./Gofirst_restart_again/sln2_bias.pkl')
            self.first_model.trans_1.weight.data=load_pickle('./Gofirst_restart/trans_1_weight.pkl')
                                                               #         print(self.first_model.trans_1.weight)
            self.first_model.trans_2.weight.data=load_pickle('./Gofirst_restart/trans_2_weight.pkl')
       #     self.first_model.trans_3.weight.data=load_pickle('./Gofirst_restart_again/trans_3_weight.pkl')
            print('first model loaded')
            
                
        #    ddd=[]
    #        for n,p in self.first_model.named_parameters():
     #           if p.requires_grad:
      #              ddd.append(n)
       #     print(len(ddd))
            #self.first_model.sln1.
           # print(self.first_model.sln1.weight.requires_grad)

        
        #*****
        self.first_model=self.first_model.to(args.gpu)
        
    
        

        #

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()

#若需要恢复间断的训练,则改这里:
#        ckpt_path = "10_pretrain_link.pth"
 #       self.load_checkpoint(ckpt_path)
       # self.model = self.model.to(args.gpu)

        if args.multiGPU and not args.inference:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()    #这里要改

            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu])
                self.first_model=DDP(self.first_model, device_ids=[args.gpu])
                
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        self.val_list=val_list

        do=0
        if do==1 and self.verbose:
            torch.save(self.first_model.state_dict(),"Gore_first.pth")

            for ig in range(32):
                save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./Gore_pkl/Gosqa_{}_{}.pkl".format(ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./Gore_pkl/Goska_{}_{}.pkl".format(ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./Gore_pkl/Gosva_{}_{}.pkl".format(ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./Gore_pkl/Gosoa_{}_{}.pkl".format(ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./Gore_pkl/Gosqb_{}_{}.pkl".format(ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./Gore_pkl/Goskb_{}_{}.pkl".format(ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./Gore_pkl/Gosvb_{}_{}.pkl".format(ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./Gore_pkl/Gosob_{}_{}.pkl".format(ig,self.args.lr))
            save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./Gore_pkl/Goslm_a_{}.pkl".format(self.args.lr))
            save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./Gore_pkl/Goslm_b_{}.pkl".format(self.args.lr))
            print('Done')


    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            project_name = "P5_Pretrain"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        #global_step = 0
        for epoch in range(self.args.epoch):
            global_step=0
     #       if epoch==0:#
        

            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)    #keep in mind this

            # Train
            self.model.train()     # 这里设置好了
            self.first_model.train()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}    #per-epoch的
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):   #step_i可用来操作acc_step
                torch.cuda.empty_cache()
               # if self.first_model.module.trans_1.weight.grad is not None:
                #    print(self.first_model.module.trans_1.weight.grad[0][:5])
                 #   print(self.first_model.module.trans_1.weight[0][:5])
         
                
                dist.barrier()

                if self.args.fp16 and _use_native_amp:
                    pass
                else:
                    if self.args.distributed:
                        #
                        dddd = next(self.model.parameters()).device

                        input_ids = batch['input_ids'].to(dddd)
                        lm_labels = batch["target_ids"].to(dddd)
                        attention_mask=batch['attn_mask'].to(dddd)

                        loss_weights = batch["loss_weights"].to(dddd)
                        B, L = lm_labels.size()

                        embeds = self.first_model(  #这里本质也是调用forward
                            input_ids=input_ids
 #                           labels=lm_labels
                        )
                        output=self.model(inputs_embeds=embeds,attention_mask=attention_mask,labels=lm_labels)

                        lm_mask = lm_labels[:,1:] != -100
                        lm_mask = lm_mask.float()

                        loss = output['loss']

                        loss = loss.view(B, L-1) * lm_mask   #注意一下loss的size

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
                torch.cuda.empty_cache()
                
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
                            torch.nn.utils.clip_grad_norm_(self.first_model.parameters(), self.args.clip_grad_norm)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()    #梯度更新

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
 
                    for param in self.model.parameters():    
                        param.grad = None
                    for param in self.first_model.parameters():    
                        param.grad = None

                global_step += 1
                if epoch==0 and global_step in [-100]:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"s_sp1_{}_8_{}_{}_mid_n_{}.pth".format(self.args.lr,self.args.train,self.args.weight_decay,global_step))
                if global_step==len(self.train_loader)//8:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Gosqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Goska_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Gosva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Gosoa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Gosqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Goskb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Gosvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Gosob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid1/Goslm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid1/Goslm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gosfirst_s_{}_{}_8_{}_mmid1.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)//4:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid1/Gosqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid1/Goska_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid1/Gosva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid1/Gosoa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid1/Gosqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid1/Goskb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid1/Gosvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid1/Gosob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid1/Goslm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid1/Goslm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gosfirst_s_{}_{}_8_{}_mid1.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*3//8:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Gosqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Goska_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Gosva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Gosoa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Gosqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Goskb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Gosvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Gosob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid2/Goslm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid2/Goslm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gosfirst_s_{}_{}_8_{}_mmid2.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)//2:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid2/Gosqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid2/Goska_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid2/Gosva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid2/Gosoa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid2/Gosqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid2/Goskb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid2/Gosvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid2/Gosob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid2/Goslm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid2/Goslm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gosfirst_s_{}_{}_8_{}_mid2.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*5//8:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Gosqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Goska_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Gosva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Gosoa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Gosqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Goskb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Gosvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Gosob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid3/Goslm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid3/Goslm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gosfirst_s_{}_{}_8_{}_mmid3.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*3//4:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid3/Gosqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid3/Goska_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid3/Gosva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid3/Gosoa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid3/Gosqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid3/Goskb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid3/Gosvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid3/Gosob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid3/Goslm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid3/Goslm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gosfirst_s_{}_{}_8_{}_mid3.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*7//8:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mend/Gosqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mend/Goska_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mend/Gosva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mend/Gosoa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mend/Gosqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mend/Goskb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mend/Gosvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mend/Gosob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mend/Goslm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mend/Goslm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gosfirst_s_{}_{}_8_{}_mend.pth".format(epoch+1,self.args.lr,self.args.train))

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
                for ig in range(32):
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_end/Gosqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_end/Goska_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_end/Gosva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_end/Gosoa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_end/Gosqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_end/Goskb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_end/Gosvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_end/Gosob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_end/Goslm_a_{}.pkl".format(epoch+1,self.args.lr))
                save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_end/Goslm_b_{}.pkl".format(epoch+1,self.args.lr))

                torch.save(self.first_model.state_dict(),"Gosfirst_s_{}_{}_8_{}_end.pth".format(epoch+1,self.args.lr,self.args.train))
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
        for epoch in range(2,8*self.args.epoch):
            torch.cuda.empty_cache()
            
            if (epoch+1)%8==1:
                doc_prefix='./llama_{}_mmid1/'.format(epoch//8+1)
                ckpt_path = "Gosfirst_s_{}_{}_8_{}_mmid1.pth".format(epoch//8+1,self.args.lr,self.args.train) 
            elif (epoch+1)%8==2:
                doc_prefix='./llama_{}_mid1/'.format(epoch//8+1)
                ckpt_path = "Gosfirst_s_{}_{}_8_{}_mid1.pth".format(epoch//8+1,self.args.lr,self.args.train) 
            elif (epoch+1)%8==3:
                doc_prefix='./llama_{}_mmid2/'.format(epoch//8+1)
                ckpt_path = "Gosfirst_s_{}_{}_8_{}_mmid2.pth".format(epoch//8+1,self.args.lr,self.args.train) 
            elif (epoch+1)%8==4:
                doc_prefix='./llama_{}_mid2/'.format(epoch//8+1)
                ckpt_path = "Gosfirst_s_{}_{}_8_{}_mid2.pth".format(epoch//8+1,self.args.lr,self.args.train)
            elif (epoch+1)%8==5:
                doc_prefix='./llama_{}_mmid3/'.format(epoch//8+1)
                ckpt_path = "Gosfirst_s_{}_{}_8_{}_mmid3.pth".format(epoch//8+1,self.args.lr,self.args.train)
            elif (epoch+1)%8==6:
                doc_prefix='./llama_{}_mid3/'.format(epoch//8+1)
                ckpt_path = "Gosfirst_s_{}_{}_8_{}_mid3.pth".format(epoch//8+1,self.args.lr,self.args.train)
            elif (epoch+1)%8==7:
                doc_prefix='./llama_{}_mend/'.format(epoch//8+1)
                ckpt_path = "Gosfirst_s_{}_{}_8_{}_mend.pth".format(epoch//8+1,self.args.lr,self.args.train)
            else:
                doc_prefix='./llama_{}_end/'.format(epoch//8+1)
                ckpt_path = "Gosfirst_s_{}_{}_8_{}_end.pth".format(epoch//8+1,self.args.lr,self.args.train)

            #
##            doc_prefix='./Gore_pkl/'
 ##           ckpt_path='Gore_first.pth'

#load model的程序在这里改:
            self.load_checkpoint(ckpt_path)
            self.first_model=self.first_model.to(self.args.gpu)

            for gg in range(32):
                self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gosqa_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Goska_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gosva_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gosoa_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gosqb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Goskb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gosvb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gosob_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
            self.model.base_model.model.lm_head.lora_A.default.weight.data=load_pickle(doc_prefix+"Goslm_a_{}.pkl".format(self.args.lr)).data.to(args.gpu)
            self.model.base_model.model.lm_head.lora_B.default.weight.data=load_pickle(doc_prefix+"Goslm_b_{}.pkl".format(self.args.lr)).data.to(args.gpu)
            if self.verbose:
                print('Main model loaded.')
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
                acc_file=open('GosLlama_s_1e-4.txt','a')                           #文件名字注意一下
                if (epoch+1)%8==1:
                    acc_file.write(str(epoch//8+1)+'_mmid1'+'\n')
                elif (epoch+1)%8==2:
                    acc_file.write(str(epoch//8+1)+'_mid1'+'\n')
                elif (epoch+1)%8==3:
                    acc_file.write(str(epoch//8+1)+'_mmid2'+'\n')
                elif (epoch+1)%8==4:
                    acc_file.write(str(epoch//8+1)+'_mid2'+'\n')
                elif (epoch+1)%8==5:
                    acc_file.write(str(epoch//8+1)+'_mmid3'+'\n')
                elif (epoch+1)%8==6:
                    acc_file.write(str(epoch//8+1)+'_mid3'+'\n')
                elif (epoch+1)%8==7:
                    acc_file.write(str(epoch//8+1)+'_mend'+'\n')
                else:
                    acc_file.write(str(epoch//8+1)+'_end'+'\n')
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
        self.first_model.eval()
        with torch.no_grad():
            for step_i, batch in tqdm(enumerate(self.val_loader)):   #每张卡单独inference完毕后per_inference只做一次结果同步
                torch.cuda.empty_cache()

                if self.args.distributed:
                    device = next(self.model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    embeds = self.first_model(input_ids=input_ids)
                    attention_mask=batch['attn_mask'].to(device)
                    results = self.model.g_step(in_embeds=embeds, attention_mask=attention_mask)

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
            'link':[['1-1-1-1','1-1-1-2'],['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4'],['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4']],
            'classification':[['2-1-1-1','2-1-1-2'],['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4'],['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4']]
            }
        elif args.train=='Arxiv':
            train_task_list = {
            'classification':[['6-6-6-7'],['2-1-1-2','2-3-1-2'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']],
            'link':[['1-1-3-1']]
     ##       'link':[['1-1-1-1','1-1-1-2','1-3-1-1','1-3-1-2'],['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4','1-3-2-1','1-3-2-2','1-3-2-3','1-3-2-4'],['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4','1-3-3-1','1-3-3-2','1-3-3-3','1-3-3-4']]
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
            ##3###3#33'link':[['1-1-1-1','1-1-1-2'],['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4'],['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4']],
            'classification':[['2-1-1-1','2-1-1-2'],['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4'],['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4']]
            }
        elif args.valid=='Arxiv':
            val_task_list = {
            #'classification':[['5-5-5-5','6-6-6-6'],['2-1-1-1','2-1-1-2','2-3-1-1','2-3-1-2'],['2-1-2-1','2-1-2-2','2-1-2-3','2-1-2-4','2-3-2-1','2-3-2-2','2-3-2-3','2-3-2-4'],['2-1-3-1','2-1-3-2','2-1-3-3','2-1-3-4','2-3-3-1','2-3-3-2','2-3-3-3','2-3-3-4']]
            'classification':[['6-6-6-7']]#,['2-3-2-4'],['2-3-3-4']]#,['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']]
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
