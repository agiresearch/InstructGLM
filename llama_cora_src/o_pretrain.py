import collections
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

from param import parse_args
from pretrain_data import get_loader#, len_val
from utils import LossMeter
from dist_utils import reduce_dict, new_reduce_dict

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)   #实际上这里传进去的train没有用

        assert args.whole_word_embed
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



        if 'p5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)   #这个我肯定要用

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            self.start_epoch = int(args.load.split('Epoch-')[-1])

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu])
                                 #find_unused_parameters=True
                                 #)
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:   
            results = self.evaluate_epoch()   

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            if 't5' in self.args.backbone:
                project_name = "P5_Pretrain"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epoch):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)    #keep in mind this

            # Train
            self.model.train()     # 这里设置好了

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        #
                        dddd = next(self.model.parameters()).device

                        input_ids = batch['input_ids'].to(dddd)
                        whole_word_ids = batch['whole_word_ids'].to(dddd)
                        lm_labels = batch["target_ids"].to(dddd)

                        loss_weights = batch["loss_weights"].to(dddd)
                        B, L = lm_labels.size()

                        output = self.model(  #这里本质也是调用forward
                            input_ids=input_ids,
                            whole_word_ids=whole_word_ids,
                            labels=lm_labels,
                            return_dict=True
                        )

                        lm_mask = lm_labels != -100
                        lm_mask = lm_mask.float()

                        loss = output['loss']

                        loss = loss.view(B, L) * lm_mask   #注意一下loss的size

                        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)   #这里之后变1维的了

                        task_counts = {task: 0 for task in self.model.module.losses}
                        task_loss = {task: 0 for task in self.model.module.losses}

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

                loss = results['loss']
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
  ###              for pp in self.model.named_parameters():
    ###                if pp[0] in ['module.shared.weight']:
     ##                   print()
       ##3                 print('seeeeeeeeeee')
          ###              print(pp[1].grad[586][:3])
             ###           print(pp[1][586][:3])
                ###        print()
                ###dist.barrier()

                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():    #本project 标准写法
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results, average=False)    #Global信息拿到


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

    
            if epoch != -1:    #这里返回接口要改
                # Validation
                valid_results = self.evaluate_epoch().to('cuda')

                valid_results = new_reduce_dict(valid_results, num_G=args.world_size)     #对一个1维的tensor标量做reduce  (Size([1]))
                dist.barrier()

                if self.verbose:
                    print()
                    print('Accuray after {} Epoch is : {}'.format(epoch+1, valid_results.sum().item()/len(self.val_loader.dataset)))
                    print()

                dist.barrier()
            


    def evaluate_epoch(self):    #关键代码#
        epoch_results = []
        self.model.eval()
        with torch.no_grad():
            observe_generation=[]
            for step_i, batch in tqdm(enumerate(self.val_loader)):
                device=batch['loss_weights'][0].device
                ttt=0   #我这里是per batch的记录

                if self.args.distributed:
                    results = self.model.module.g_step(batch)
                else:
                    results = self.model.g_step(batch)   #results为一个长度为B的list,元素类型为str

               

                for iiid in range(len(results)):    #这里后期引入其他任务后还要改
                    focus=results[iiid].split(' ')[0]    #这里需要检查的
                    if batch['temp_ids'][iiid] in ['6-1','6-2']:
                        if focus==batch['target_text'][iiid]:
                            ttt+=1
                            observe_generation.append('first')
                            observe_generation.append(focus)
                    elif batch['temp_ids'][iiid] in ['6-3','6-4']:
                        if focus.lower() in ('yes', 'true', 't', 'y', '1'):
                            ff=True
                        elif focus.lower() in ('no', 'false', 'f', 'n', '0'):
                            ff=False
                        else:
                            ff=focus
                        fff=True if batch['target_text'][iiid]=='yes' else False

                        if ff==fff:
                            observe_generation.append('second')
                            observe_generation.append(focus)
                            ttt+=1
                epoch_results.append(ttt)  #我这里就是记录了一个（该gpu的该Iteration部分的）总数
                dist.barrier()

            omg=0
            if self.args.gpu==0 and omg==1:
                print()
                print('seeeeeeeeeee')
                print()
                print(observe_generation)
                print()
                print(len(observe_generation))
                temp_count=0
                for ccccc in observe_generation:
                    if ccccc=='first':
                        temp_count+=1
                print(temp_count)
                print()
            dist.barrier()

            return torch.Tensor([sum(epoch_results)]).to(device)   #我这个sum外面的[]加的很重要
        
    









    def old_evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=275)

            for step_i, batch in enumerate(self.val_loader):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:      #只打印了主GPU的loss,而且没有metric,意义不大
                    desc_str = f'Valid Epoch {epoch} |'
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)    #这里打印
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()
            dist.barrier()

            return epoch_results


def main_worker(gpu, args):     # the gpu is the args.local_rank
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    # define the prompts used in training
    if args.train == 'yelp':
        train_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
        'review': ['4-1', '4-2'],
        'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
        }
    else:
        train_task_list = {#'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        #'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        #'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
        #'review': ['4-1', '4-2', '4-3'],
        #'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7'],
        'link':['6-1','6-2','6-3','6-4']
        }
    # define sampling numbers for each group of personalized prompts (see pretrain_data.py)
    # if greater than 1, a data sample will be used for multiple times （with different prompts） （in certain task family）
    train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1, 'traditional': (10, 5), 'link': 2}
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

    print(f'Building val loader at GPU {gpu}')
    # define the prompts used in validation
    if args.valid == 'yelp':
        val_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
        'review': ['4-1', '4-2'],
        'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
        }
    else:
        val_task_list = {#'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        #'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        #'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
        #'review': ['4-1', '4-2', '4-3'],
        #'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7'],
        'link':['6-1','6-2','6-3','6-4']
        }
    val_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1), 'link': 1}
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

    trainer = Trainer(args, train_loader, val_loader, train=True)    # Remember to set 'train' = True
    trainer.train()


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
    if 'yelp' in args.train:
        dsets.append('yelp')
    comments.append(''.join(dsets))
    if args.backbone:
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
