import sys,time
import numpy as np
import torch
import os
import logging
import glob
import math
import json
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
from collections import Counter,defaultdict
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils
# from seqeval.metrics import classification_report # Commented as it does not seem to be used
import torch.nn.functional as F
import nlp_data_utils as data_utils
from copy import deepcopy
sys.path.append("./approaches/base/")
from .bert_adapter_base import Appr as ApprBase
from .my_optimization import BertAdam

class Appr(ApprBase):


    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('BERT ADAPTER AdaBOP')
        
        # self.fea_in_hook = {} # MS: Not sure where this is used
        self.fea_in = defaultdict(dict)
        # self.fea_in_count = defaultdict(int) # MS: Not sure where this is used

        return

    def train(self,t,train,valid,args,num_train_steps,save_path,train_data,valid_data):

        print('We are at task :',t)
        
        global_step = 0
        self.model.to(self.device)

        t_total = num_train_steps
        # if t==0:
        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'svd': True, 'lr': self.args.learning_rate, 'thres': self.args.svd_thres},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'svd': True, 'lr': self.args.learning_rate, 'thres': self.args.svd_thres}
            ]
        # self.model_optimizer = BertAdam(optimizer_grouped_parameters,
                             # lr=self.args.learning_rate,
                             # warmup=self.args.warmup_proportion,
                             # t_total=t_total)
        self.model_optimizer = Adam(optimizer_grouped_parameters,lr=self.args.learning_rate)
        
        if t>0 and len(self.fea_in)==0: # Only need to do this when loading from checkpoint to continue training
            # self.model_optimizer.load_state_dict(self.model_optimizer_state_dict)
            # self.model_optimizer.transforms = self.model_optimizer_transforms
            i = -1
            for group in self.model_optimizer.param_groups:
                svd = group['svd']
                # print(svd)
                if svd is False:
                    continue
                for p in group['params']:
                    if p.requires_grad is False:
                        continue
                    i += 1
                    self.fea_in[p] = self.prev_task_fea_in[i]
            # print('\nCheck fea_in',len(self.fea_in))

        
        all_targets = []
        for step, batch in enumerate(train):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, tasks= batch
            all_targets += list(targets.cpu().numpy())
        class_counts_dict = dict(Counter(all_targets))
        class_counts = [class_counts_dict[k] for k in np.unique(all_targets)] # .unique() returns ordered list
        
        all_targets = []
        for step, batch in enumerate(valid):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, tasks= batch
            all_targets += list(targets.cpu().numpy())
        class_counts_dict = dict(Counter(all_targets))
        valid_class_counts = [class_counts_dict[k] for k in np.unique(all_targets)]

        
        # get correlation
        # self.tc_lamb = {}
        self.tc_lamb = []
        if t == 0:
            self.grad = self.train_task_correlation(train,valid,class_counts)
            # for i in range(len(self.grad)):
                # self.tc_lamb.append(1.0) # Set to 1 for first task
            # for p in self.grad.keys():
                # self.tc_lamb[p] = 1.0 # Set to 1 for first task
        else:
            grad_pre = deepcopy(self.grad) # This cuases keyerror for grad_pre[p]
            # grad_pre = [self.grad[p] for p in self.grad.keys()]            
            self.grad = self.train_task_correlation(train,valid,class_counts)

            # print('grad:',grad[0].shape)
            # print('grad_pre:',grad_pre[0].shape)
            tc_vals = []
            for i in range(len(self.grad)):
            # for p in self.grad.keys():
                grad_norm = np.linalg.norm(self.grad[i])
                # print(type(grad_pre),len(grad_pre))
                # print(type(self.grad),len(self.grad))
                # print(grad_pre[i])
                # print(grad_pre[i].T)
                projection = np.dot(self.grad[i],grad_pre[i].T)
                projection_norm = np.linalg.norm(projection)
                tc_vals.append(projection_norm/grad_norm)
                if projection_norm > self.args.tc_epsilon * grad_norm:
                    # self.tc_lamb[p] = self.args.tc_lamb_s
                    self.tc_lamb.append(self.args.tc_lamb_s)
                else:
                    # self.tc_lamb[p] = self.args.tc_lamb_l
                    self.tc_lamb.append(self.args.tc_lamb_l)
            # calc projections
            # with torch.no_grad(): # This doesn't makse sense. get_eigens() and get_transforms() require grad to exist
            print('TC Hist:',np.histogram(tc_vals))
            np.save(self.args.my_save_path+'tc_vals.npy',tc_vals)
            self.update_optim_transforms()


        best_loss=np.inf
        best_model=utils.get_model(self.model)
        patience=self.args.lr_patience

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar,t_total,global_step,class_counts)
            clock1=time.time()

            train_loss=self.eval_validation(t,train,class_counts)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f} |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss),end='')

            valid_loss=self.eval_validation(t,valid,valid_class_counts)
            print(' Valid: loss={:.3f} |'.format(valid_loss),end='')
            # Adapt lr
            if best_loss-valid_loss > args.valid_loss_es:
                patience=self.args.lr_patience
                # print(' *',end='')
            else:
                patience-=1
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                print(' *',end='')
            if patience<=0:
                break

            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)
        
        # Save model
        # torch.save(self.model.state_dict(), save_path+str(args.note)+'_seed'+str(args.seed)+'_model'+str(t))
        
        # Calc and save fea to use to calc projections (0transforms) for subsequent task
        self.calc_fea(train)
        
        return

    def train_epoch(self,t,data,iter_bar,t_total,global_step,class_counts):
        self.num_labels = self.taskcla[t][1]
        self.model.train()
        
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, tasks= batch

            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            # Forward
            if 'dil' in self.args.scenario:
                outputs=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                # output = outputs[t]
            elif 'cil' in self.args.scenario:
                outputs=output_dict['y']
            loss=self.criterion_train(tasks,outputs,targets,class_counts)

            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in self.model_optimizer.param_groups:
                param_group['lr'] = lr_this_step
            self.model_optimizer.step()
            self.model_optimizer.zero_grad()
            global_step += 1

        return global_step

    def eval(self,t,data,test=None,trained_task=None):
        total_loss=0
        total_acc=0
        total_num=0
        target_list = []
        pred_list = []


        with torch.no_grad():
            self.model.eval()

            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, tasks= batch
                real_b=input_ids.size(0)

                output_dict = self.model.forward(input_ids, segment_ids, input_mask)
                # Forward
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]
                elif 'cil' in self.args.scenario:
                    output=output_dict['y']
                
                if 'cil' in self.args.scenario and self.args.use_rbs:
                    loss=self.ce(t,output,targets)
                else:
                    loss=self.ce(output,targets)

                _,pred=output.max(1)
                hits=(pred==targets).float()

                target_list.append(targets)
                pred_list.append(pred)

                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b

            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')

                # break

        return total_loss/total_num,total_acc/total_num,f1
    
    def eval_validation(self,_,data,class_counts):
        total_loss=0
        total_num=0
        self.model.eval()
        with torch.no_grad():
            # Loop batches
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, tasks= batch
                real_b=input_ids.size(0)

                output_dict = self.model.forward(input_ids, segment_ids, input_mask)
                outputs = output_dict['y']

                # Forward
                loss=self.criterion_train(tasks,outputs,targets,class_counts)

                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_num+=real_b


        return total_loss/total_num

    def criterion_train(self,tasks,outputs,targets,class_counts):
        loss=0
        # loss2=0
        for t in np.unique(tasks.data.cpu().numpy()):
            t=int(t)
            # output = outputs  # shared head

            if 'dil' in self.args.scenario:
                output=outputs #always shared head
            elif 'til' in self.args.scenario:
                output = outputs[t]
            elif 'cil' in self.args.scenario:
                output = outputs

            idx=(tasks==t).data.nonzero().view(-1)
            # print('Debugging:',output.shape,output[0])
            # print('Debugging:',targets.shape,targets[0])
            if 'cil' in self.args.scenario and self.args.use_rbs:
                loss+=self.ce(t,output[idx,:],targets[idx],class_counts)*len(idx)
            else:
                # print(output[idx,:],targets[idx])
                loss+=self.ce(output[idx,:],targets[idx])*len(idx)
            
            # loss2+=self.ce2(output[idx,:],targets[idx])*len(idx)
        # try:
            # assert loss.item()==loss2.item()
        # except AssertionError:
            # print(loss.item(),loss2.item()) #TODO: Check why there is variation after the 4th decimal
        
        return loss/targets.size(0)

    def update_optim_transforms(self):
                   
        self.model_optimizer.get_eigens(self.fea_in, self.tc_lamb)
        time_svd_start = time.time()
        self.model_optimizer.get_transforms()
        time_svd_end = time.time()
        time_svd = time_svd_end - time_svd_start
        print('Time for updating the Orthogonal Projection:  ', time_svd)
    
    def calc_fea(self, train_loader):
        modules = [m for n, m in self.model.named_modules() if hasattr(
            m, 'weight')] # and not bool(re.match('last', n))]
        handles = []
        for m in modules:
            handles.append(m.register_forward_hook(hook=self.compute_cov))
        
        for step, batch in enumerate(train_loader):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, tasks= batch

            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            # sys.exit()
            if 'dil' in self.args.scenario:
                outputs=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                # output = outputs[t]
            elif 'cil' in self.args.scenario:
                outputs=output_dict['y']
            loss=self.criterion_train(tasks,outputs,targets,class_counts=None) # MS: We don't actually need the right loss or grad values
            
            self.model_optimizer.zero_grad()
            # self.model_scheduler.step(epoch)
            loss.backward()

        self.model_optimizer.zero_grad()
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()
    
    def compute_cov(self, module, fea_in, fea_out):
        if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm): # MS: Include layernorm
            # print(module, len(fea_in), fea_in[0].shape)
            self.update_cov(torch.squeeze(torch.mean(fea_in[0], 0, True)), module.weight) # Take mean of all samples in batch, MS: Then, remove the first dimension with size=1

        # elif isinstance(module, nn.Conv2d):
            # kernel_size = module.kernel_size
            # stride = module.stride
            # padding = module.padding

            
            # fea_in_ = F.unfold(
                # torch.mean(fea_in[0], 0, True), kernel_size=kernel_size, padding=padding, stride=stride)

            # fea_in_ = fea_in_.permute(0, 2, 1)
            # fea_in_ = fea_in_.reshape(-1, fea_in_.shape[-1])
            # self.update_cov(fea_in_, module.weight)
        
        else:
            print('comput_cov not implemented:',module) # Checked that this consists of only the Embedding layer

        torch.cuda.empty_cache()
        return None

    def update_cov(self, fea_in, k):
        if len(fea_in.shape)==1: fea_in = torch.stack([fea_in]) # MS: Handle cases with single dimension
        cov = torch.mm(fea_in.transpose(0, 1), fea_in)
        if len(self.fea_in[k]) == 0:
            self.fea_in[k] = cov
        else:
            self.fea_in[k] = self.fea_in[k] +  cov
    
    def train_task_correlation(self,train_loader,val_loader=None,class_counts=None):
        self.model.train()
        for step, batch in enumerate(train_loader):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, tasks= batch

            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            # Forward
            if 'dil' in self.args.scenario:
                outputs=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                # output = outputs[t]
            elif 'cil' in self.args.scenario:
                outputs=output_dict['y']
            loss=self.criterion_train(tasks,outputs,targets,class_counts)
            
            self.model_optimizer.zero_grad()
            # self.model_scheduler.step(epoch)
            loss.backward()
        grad_list = self.model_optimizer.get_correlation()
        # for k in grad_list.keys():
            # grad_list[k] /= (step+1) # MS: Average the gradients
        return grad_list

    
# source: https://github.com/hyscn/AdaBOP/blob/main/optim/adam_svd.py
class Adam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, svd=False, thres=1.001,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, svd=svd,
                        thres=thres)
        super(Adam, self).__init__(params, defaults)

        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('svd', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # no_adabop_params = 0
        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)
                # print('Checking .step():',svd,len(self.transforms))
                if svd and len(self.transforms) > 0:
                    if len(update.shape) == 4:
                        # the transpose of the manuscript
                        update_ = torch.mm(update.view(update.size(
                            0), -1), self.transforms[p]).view_as(update)
                       
                    else:
                        if self.transforms[p] is not None: # For the params where we don't calc COV in compute_cov()
                            # print(update.shape, self.transforms[p].shape)
                            update_ = torch.matmul(update, self.transforms[p]) # torch.mm(update, self.transforms[p])
                        else:
                            # print('self.transforms[p] is None')
                            # no_adabop_params += 1
                            update_ = update
                        
                else:
                    update_ = update
                p.data.add_(update_)
        # print('No Adabop update for :',no_adabop_params)
        return loss

    def get_correlation(self,closure = None):
        # grad_list = {}
        grad_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.detach().cpu().numpy()
                grad = grad.reshape(grad.shape[0],-1)
                grad_list.append(grad)
                # grad_list[p] = grad
        return grad_list

    def get_transforms(self):
        # print(len(self.param_groups))
        for group in self.param_groups:
            svd = group['svd']
            # print(svd)
            if svd is False:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                thres = group['thres']
                if self.eigens[p]['eigen_value'] is None: # or thres<0.01:  # For the params where we don't calc COV in compute_cov() (OR) thres is too low
                    # print('skipping this p')
                    self.transforms[p] = None
                    continue
                # ind = self.eigens[p]['eigen_value'] <= self.eigens[p]['eigen_value'][-1] * thres # MS: Does not work - will choose only the last eigen value always, or None if thres<1?!
                thres = int(self.eigens[p]['eigen_value'].shape[0] * thres) # 20 = 100 * 0.2
                # ind = [True if i < thres else False for i in range(self.eigens[p]['eigen_value'].shape[0])]
                thres = self.eigens[p]['eigen_value'].shape[0] - thres # 80 = 100 - 20
                ind = [True if i >= thres else False for i in range(self.eigens[p]['eigen_value'].shape[0])]  # MS: Take bottom-K eigen values (corresponding to null space) using 0<thres<=1
                ind = torch.tensor(ind)
                print('reserving basis {}/{}; cond: {}, radio:{}'.format(
                    ind.sum(), self.eigens[p]['eigen_value'].shape[0],
                    self.eigens[p]['eigen_value'][0] /
                    self.eigens[p]['eigen_value'][-1],
                    self.eigens[p]['eigen_value'][ind].sum(
                    ) / self.eigens[p]['eigen_value'].sum()
                ))
                # GVV^T
                # get the columns
                basis = self.eigens[p]['eigen_vector'][:, ind]
                transform = torch.mm(basis, basis.transpose(1, 0))
                self.transforms[p] = transform / torch.norm(transform)
                self.transforms[p].detach_()
                # print(self.transforms[p].shape, self.transforms[p].min(), self.transforms[p].max())
        print(len(self.transforms))
        # sys.exit()

    def get_eigens(self, fea_in, tc_lamb): #MS: passing tc_lamb from outside
        i, excl_params = -1, 0
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                i += 1
                eigen = self.eigens[p]
                device=torch.device("cuda")
                # _, eigen_value, eigen_vector = torch.svd(fea_in[p] + 0.0075 * torch.eye(fea_in[p].size(0)).cuda())
                if fea_in[p]=={}: # Presumably for the params where we don't calc COV in compute_cov()
                    excl_params += 1
                    eigen['eigen_value'],eigen['eigen_vector'] = None, None
                else:
                    # print('At param ',i,' fea_in:',tc_lamb[i],fea_in[p],'\n')
                    _, eigen_value, eigen_vector = torch.svd(tc_lamb[i]*fea_in[p] + torch.eye(fea_in[p].size(0)).cuda()) # MS: Above original line looks wrong compared to eq 26 of paper
                    eigen['eigen_value'] = eigen_value
                    eigen['eigen_vector'] = eigen_vector
        print('\nExcluded params where eigen not calculated:',excl_params,'\n')

    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * \
            math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update
