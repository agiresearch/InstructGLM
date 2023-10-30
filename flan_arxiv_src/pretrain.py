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
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
from param import parse_args
from pretrain_data import get_loader
from utils import LossMeter
from dist_utils import reduce_dict, new_reduce_dict

_use_native_amp = False
_use_apex = False

_use_native_amp = True
from torch.cuda.amp import autocast

from trainer_base import TrainerBase


# The Trainer inherits TrainerBase in trainer_base.py
# We didn't deploy fp16 training in our paper for Flan-T5 series backbones.
class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True, val_list=None):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)   

        from pretrain_model import InstructGLM

        model_kwargs = {}

        model_class = InstructGLM

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config, **model_kwargs)

        self.model.resize_token_embeddings(self.tokenizer.vocab_size + 169343) # Extend the vocabulary of the LLM backbone. There are 169343 nodes in Arxiv Graph.

        self.model.tokenizer = self.tokenizer

        
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()

      # For restart training if needed.
      #  ckpt_path="your_restart_checkpoint_name.pth"
      #  self.load_checkpoint(ckpt_path)

        self.model = self.model.to(args.gpu)

        # Optimizer: AdamW with weight decay at 0
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU and not args.inference:   #  DDP setup
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu])

        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        self.val_list=val_list

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME  

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            project_name = "Natural Language is All a Graph Needs"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        for epoch in range(self.args.epoch):
            global_step=0

            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)   

            # Train
            self.model.train()    

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}    # per-epoch
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):   
                dist.barrier()

                if self.args.fp16 and _use_native_amp:
                    pass
                else:
                    if self.args.distributed:
                        dddd = next(self.model.parameters()).device

                        input_ids = batch['input_ids'].to(dddd)
                        lm_labels = batch["target_ids"].to(dddd)

                        loss_weights = batch["loss_weights"].to(dddd)
                        B, L = lm_labels.size()
                         # forward
                        output = self.model( 
                            input_ids=input_ids,
                            real_feature=self.train_loader.dataset.real_feature.to(dddd), # The previous part of real_feature are all zero vectors.  
                            labels=lm_labels,
                            return_dict=True
                        )

                        lm_mask = lm_labels != -100
                        lm_mask = lm_mask.float()

                        loss = output['loss']

                        loss = loss.view(B, L) * lm_mask  

                        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)   

                        results = {}     

                        results['loss'] = (loss * loss_weights).mean()   
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
                        self.optim.step()    # Update

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
 
                    for param in self.model.parameters():    
                        param.grad = None

                global_step += 1
                
                if global_step==len(self.train_loader)//2:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"GL_flan_{}_{}_8_{}_mid2.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)//4:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"GL_flan_{}_{}_8_{}_mid1.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*3//4:
                    if self.verbose:
                        torch.save(self.model.state_dict(),"GL_flan_{}_{}_8_{}_mid3.pth".format(epoch+1,self.args.lr,self.args.train))

                dist.barrier()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    lr=self.optim.param_groups[-1]['lr']

                for k, v in results.items():   
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 1==0:      # Logging purpose
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'   

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:   
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']   
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results,average=False)    # Get Global information

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
                print(losses_str)     # Print once per epoch

            dist.barrier()

            if self.verbose:  # Save checkpoint
                torch.save(self.model.state_dict(),"GL_flan_{}_{}_8_{}_end.pth".format(epoch+1,self.args.lr,self.args.train))

            dist.barrier()



    def test(self):   
        for epoch in range(4*self.args.epoch):
            if (epoch+1)%4==1:
                ckpt_path = "GL_flan_{}_{}_8_{}_mid1.pth".format(epoch//4+1,self.args.lr,self.args.train) 
            elif (epoch+1)%4==2:
                ckpt_path = "GL_flan_{}_{}_8_{}_mid2.pth".format(epoch//4+1,self.args.lr,self.args.train) 
            elif (epoch+1)%4==3:
                ckpt_path = "GL_flan_{}_{}_8_{}_mid3.pth".format(epoch//4+1,self.args.lr,self.args.train) 
            else:
                ckpt_path = "GL_flan_{}_{}_8_{}_end.pth".format(epoch//4+1,self.args.lr,self.args.train)
            
            #One can directly assign the checkpoint here when testing.
            #ckpt_path='xxx.pth'

            self.load_checkpoint(ckpt_path)
            self.model = self.model.to(self.args.gpu)
            
            valid_results = self.evaluate_epoch()    # For accuracy
            dist.barrier()

            valid_results = new_reduce_dict(valid_results)   
            dist.barrier()

            if self.verbose:
                print()
                print()
                for kk in valid_results.keys():
                    if kk.endswith('transductive'):
                        if self.args.train=='Arxiv':
                            valid_results[kk]=valid_results[kk].item() / self.val_loader.dataset.len_transductive
                print(valid_results)
                print()
                print()

            dist.barrier()

            if self.verbose:
                acc_file=open('GLM_Flan_t5_Large.txt','a')                     
                if (epoch+1)%4==1:
                    acc_file.write(str(epoch//4+1)+'_mid1'+'\n')
                elif (epoch+1)%4==2:
                    acc_file.write(str(epoch//4+1)+'_mid2'+'\n')
                elif (epoch+1)%4==3:
                    acc_file.write(str(epoch//4+1)+'_mid3'+'\n')
                else:
                    acc_file.write(str(epoch//4+1)+'_end'+'\n')

                acc_file.write(str(valid_results)+'\n\n')
                acc_file.close()
            dist.barrier()


    def evaluate_epoch(self):   
        ACC={}
        for k in list(self.val_list.keys()):
            if k=='link':
                pass
            elif k=='classification':
                if self.args.valid=='Arxiv':
                    templates=[]
                    for tems in self.val_list[k]:
                        templates=templates+tems
                    for thing in templates:
                        ACC[thing+'-'+'transductive']=0

        self.model.eval()
        with torch.no_grad():
            for step_i, batch in tqdm(enumerate(self.val_loader)):   

                if self.args.distributed:
                    results = self.model.g_step(batch,real=self.val_loader.dataset.real_feature)

                for iiid in range(len(results)):    
                    task=batch['task'][iiid]
                    temp_id=batch['temp_ids'][iiid]

                    if task=='classification':
                        cate=batch['cate'][iiid] 
                        if temp_id.endswith('2') or temp_id.endswith('4') or temp_id.endswith('6'):  
                            if results[iiid].lower() == batch['target_text'][iiid]: 
                                #Check if the generated text strings is strictly matched with the label in natural language.
                                ACC[temp_id+'-'+cate]+=1     
                        else:   
                            pass
                    elif task=='link':
                        pass

                dist.barrier()

            return ACC   


def main_worker(gpu, args):     # the gpu represents the local_rank
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    # define the prompts used in training
    if not args.inference:
        print(f'Building train loader at GPU {gpu}')    # Train Consoles

        # The '6-6-6-6' represents the graph-free instruction prompt. 
        # All detailed instruction prompt lists are summarized in all_graph_templates.py and the appendix of our paper.

        if args.train=='Arxiv':
            train_task_list = {
            'classification':[['6-6-6-6'],['2-1-1-2','2-3-1-2'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']],
            'link':[['1-1-1-1','1-1-1-2','1-3-1-1','1-3-1-2'],['1-1-2-1','1-1-2-2','1-1-2-3','1-1-2-4','1-3-2-1','1-3-2-2','1-3-2-3','1-3-2-4'],['1-1-3-1','1-1-3-2','1-1-3-3','1-1-3-4','1-3-3-1','1-3-3-2','1-3-3-3','1-3-3-4']]
            }
        else:
            pass

        train_sample_numbers = {} # Abandoned

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
        trainer = Trainer(args,train_loader= train_loader,  train=True)  
        trainer.train()


    # define the prompts used in validation/ Test
    if args.inference:
        print(f'Building val loader at GPU {gpu}')         # Valid/ Test Consoles
        # The '6-6-6-6' represents the graph-free instruction prompt. 
        if args.valid=='Arxiv':
            val_task_list = {
            #'classification':[['6-6-6-6'],['2-1-1-2','2-3-1-2'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']]
            'classification':[['2-3-1-2'],['6-6-6-6']]  
            }

        val_sample_numbers = {} # Abandoned

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
    if 'Arxiv' in args.train:
        dsets.append('Arxiv')
    comments.append(''.join(dsets))

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
