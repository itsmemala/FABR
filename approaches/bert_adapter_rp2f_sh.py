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
from collections import Counter
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR
import utils
# from seqeval.metrics import classification_report # Commented as it does not seem to be used
import torch.nn.functional as F
import nlp_data_utils as data_utils
from copy import deepcopy
sys.path.append("./approaches/base/")
from .bert_adapter_base import Appr as ApprBase
from .my_optimization import BertAdam

from captum.attr import LayerIntegratedGradients

class Appr(ApprBase):


    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('BERT ADAPTER EWC NCL')

        return

    def train(self,t,train,valid,args,num_train_steps,save_path,train_data,valid_data):

        global_step = 0
        self.model.to(self.device)

        # param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
            # {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            # {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # ]
        t_total = num_train_steps
        # optimizer = BertAdam(optimizer_grouped_parameters,
                             # lr=self.args.learning_rate,
                             # warmup=self.args.warmup_proportion,
                             # t_total=t_total)
        
        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights
        
        # Initialise aux for fisher calc
        self.aux_net = deepcopy(self.model)  # cal fisher for online model
        
        # Initialise learner
        if self.args.learner_is_seq:
            if t==0:
                self.learner=deepcopy(self.model)
            else:
                self.learner=deepcopy(self.learner_old)
        else:
            self.learner=deepcopy(self.model)
        
        if t==0:
            self.precision_matrices = {}
            for n, p in self.model.named_parameters():
                self.precision_matrices[n] = torch.zeros_like(p.data).to(self.device)
        
        cur_precision_matrices = {}
        for n, p in self.model_old.named_parameters():
            cur_precision_matrices[n] = torch.zeros_like(p.data).to(self.device)

        opt_learner = BertAdam([(k, v) for k, v in self.learner.named_parameters() if v.requires_grad==True], lr=self.args.learning_rate, warmup=self.args.warmup_proportion, t_total=t_total)
        opt_classifier = BertAdam([(k, v) for k, v in self.learner.named_parameters() if v.requires_grad==True and 'last' in k], lr=self.args.learning_rate, warmup=self.args.warmup_proportion, t_total=t_total)

        scheduler_feature = StepLR(opt_learner, step_size=99, gamma=0.1)
        scheduler_classifier = StepLR(opt_classifier, step_size=99, gamma=0.1)


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

        best_loss=np.inf
        best_model=utils.get_model(self.model)
        patience=self.args.lr_patience

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step,cur_precision_matrices=self.train_epoch(t,train,iter_bar, opt_classifier,opt_learner,scheduler_feature,scheduler_classifier,cur_precision_matrices,global_step,class_counts,train_data)
            # print('\n\n',global_step,cur_precision_matrices['bert.encoder.layer.0.attention.output.LayerNorm.weight'][:10])
            
            clock1=time.time()
            # print('\n\n',global_step,self.precision_matrices['bert.encoder.layer.0.attention.output.LayerNorm.weight'][:10])

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
        
        for n, p in self.model_old.named_parameters():
            self.precision_matrices[n] += cur_precision_matrices[n]
        
        # Restore best
        utils.set_model_(self.model,best_model)

        return

    def train_epoch(self,t,data,iter_bar,opt_classifier,opt_learner,scheduler_feature,scheduler_classifier,cur_precision_matrices,global_step,class_counts,train_data):
        self.num_labels = self.taskcla[t][1]
        self.model.train()
        
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, tasks= batch

            # =====  =====  ===== update classifier =====  =====  =====
            # with torch.no_grad():
                # feat = self.model.features(input_ids, segment_ids, input_mask)

            # outputs = self.model.classifier(feat)['y']
            # loss_ce = self.criterion_train(tasks,outputs,targets,class_counts)
            # classifier_loss = loss_ce

            # opt_classifier.zero_grad()
            # classifier_loss.backward()
            # opt_classifier.step()

            # with torch.no_grad():
                # feat = self.model.features(input_ids, segment_ids, input_mask)
            # print(feat[:10])
            # # =====  =====  ===== update target network =====  =====  =====
            for (online_params_n,online_params), (target_params_n,target_params), (old_params_n,old_params) \
                    in zip(self.learner.named_parameters(), self.model.named_parameters(),
                                     self.model_old.named_parameters()):
                online_weight, target_weight, old_weight = online_params.data, target_params.data, old_params.data
                if online_params.requires_grad==True:
                    # print(online_params_n)
                    if t == 0:
                        target_params.data = online_weight * 1.
                    else:
                        cur_fisher, old_fisher = cur_precision_matrices[online_params_n], self.precision_matrices[online_params_n]
                        cur_fisher, old_fisher = cur_fisher / (cur_fisher + old_fisher + 1e-10), old_fisher / (cur_fisher + old_fisher + 1e-10)
                        print(online_params_n,old_fisher.shape,old_fisher.min(),old_fisher.max())

                        target_params.data = old_fisher * old_weight + cur_fisher * online_weight
                        # print(online_params_n,target_params[:2])
                        # if online_params_n=='bert.encoder.layer.0.attention.output.LayerNorm.weight': print(online_params_n, old_fisher[:10], cur_fisher[:10], old_weight[:10], online_weight[:10])
                        # sys.exit()

            # with torch.no_grad():
                # feat = self.model.features(input_ids, segment_ids, input_mask)
            # print(feat[:10])
            # =====  =====  ===== update learner =====  =====  =====
            online_feat = self.learner.features(input_ids, segment_ids, input_mask)
            outputs = self.learner.classifier(online_feat)['y']
            # print(online_feat[:2],outputs[:2,:2])
            supervised_loss = self.criterion_train(tasks,outputs,targets,class_counts)

            f_map = torch.transpose(online_feat, 0, 1)
            f_map = f_map - f_map.mean(dim=0, keepdim=True)
            f_map = f_map / torch.sqrt(0.00000001 + f_map.var(dim=0, keepdim=True))

            corr_mat = torch.matmul(f_map.t(), f_map)
            loss_mu = (self.off_diagonal(corr_mat).pow(2)).mean()

            learner_loss = supervised_loss + self.args.rp2f_lamb * loss_mu
            # print(supervised_loss,loss_mu,learner_loss)

            opt_learner.zero_grad()
            learner_loss.backward(retain_graph=False)
            opt_learner.step()
            
            # with torch.no_grad():
                # feat = self.model.features(input_ids, segment_ids, input_mask)
            # print(feat[:10])
            
            global_step += 1
            # break

        scheduler_feature.step()
        scheduler_classifier.step()
        
        for online_params, aux_params in zip(self.learner.parameters(), self.aux_net.parameters()):
            aux_params.data = online_params.data
        # for (n,target_params), (_,aux_params) in zip(self.model.named_parameters(), self.aux_net.named_parameters()):
            # if 'last' in n:
                # aux_params.data = target_params.data
        # Needs to be implemeneted for BERT-Adapter
        # cur_precision_matrices = self._diag_fisher(t,train_data,self.device,self.aux_net,self.ce,scenario=self.args.scenario)
        cur_precision_matrices = self.fisher_matrix_diag_bert(t,train_data,self.device,self.aux_net,self.ce,scenario=self.args.scenario,imp=self.args.imp)
        # print('\n\n',global_step,cur_precision_matrices['bert.encoder.layer.0.attention.output.LayerNorm.weight'][:10])


        return global_step,cur_precision_matrices

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
    
    def eval_validation(self,t,data,class_counts):
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
                # print('\n\nOutputs:',outputs[:2,:])
                # with torch.no_grad():
                    # feat = self.model.features(input_ids, segment_ids, input_mask)
                # print(feat[:10])

                # Forward
                loss=self.criterion_train(tasks,outputs,targets,class_counts)

                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_num+=real_b
                # break


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
                loss+=self.ce(output[idx,:],targets[idx])*len(idx) 
            # print(loss)
            
            # loss2+=self.ce2(output[idx,:],targets[idx])*len(idx)
        # try:
            # assert loss.item()==loss2.item()
        # except AssertionError:
            # print(loss.item(),loss2.item()) #TODO: Check why there is variation after the 4th decimal
        
        return loss/targets.size(0)
    
    
    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def fisher_matrix_diag_bert(self,t,train,device,model,criterion,sbatch=20,scenario='til',imp='loss'):
        # Init
        fisher={}
        for n, p in model.named_parameters():
            # print(n)
            fisher[n]=0*p.data
            
        # Compute
        model.train()

        for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
            b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)])))#.cuda()
            batch=train[b]
            batch = [
                bat.to(device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch

            # Forward and backward
            model.zero_grad()
            output_dict=model.forward(input_ids, segment_ids, input_mask)
            if 'til' in scenario:
                outputs=output_dict['y']
                output = outputs[t]
            elif 'cil' in scenario:
                output=output_dict['y']
            elif 'dil' in scenario:
                output=output_dict['y']

            if imp=='loss':
                loss=criterion(t,output,targets) if scenario=='cil' and self.args.use_rbs else criterion(output,targets)
                loss.backward()
                # Get gradients
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n]+=sbatch*p.grad.data.pow(2)
            elif imp=='function':
                # Square of the l2-norm: output.pow(2).sum(dim=1)
                # Calculate square of the l2-norm and then sum for all samples in the batch
                output = output.pow(2).sum(dim=1).sum()
                output.backward()
                # Get gradients
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n]+=sbatch*torch.abs(p.grad.data)
            
        # Mean
        for n,_ in model.named_parameters():
            fisher[n]=fisher[n]/len(train)
            fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    
        return fisher
    
    def _diag_fisher(self, t, train,device,model,criterion,sbatch=128,scenario='til'):

        precision_matrices = {}
        for n, p in model.named_parameters():
            # print(n,p.shape)
            # precision_matrices[n] = torch.zeros_like(p.data).to(device)
            if 'bias' in n: # if len(p.shape) == 1:
                precision_matrices[n] = torch.ones_like(p.data).to(device)
            else:
                precision_matrices[n] = torch.zeros_like(p.data).to(device)
        # sys.exit()

        opt = BertAdam([(k, v) for k, v in model.named_parameters() if v.requires_grad==True], lr=self.args.learning_rate)
        # count=0
        for n, p in tqdm(model.named_parameters(),desc='Fisher diagonal',ncols=100,ascii=True): # 12 layer BERT * 6 trainable params * 2 (wgt + bias) = 144 (out of 297 named_parameters)
            if p.requires_grad==True and 'bias' not in n:
                param_w_weight = deepcopy(p.data)
                p.data = torch.ones_like(p.data).to(device) * 0.00001
                # count += 1
                for i in range(0,len(train),sbatch):
                    b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)])))#.cuda()
                    batch=train[b]
                    batch = [
                        bat.to(device) if bat is not None else None for bat in batch]
                    input_ids, segment_ids, input_mask, targets,_= batch

                    feat = model.features(input_ids, segment_ids, input_mask)
                    output = self.model.classifier(feat)['y']

                    loss = criterion(t,output,targets) if scenario=='cil' and self.args.use_rbs else criterion(output,targets)
                    opt.zero_grad()
                    loss.backward()

                precision_matrices[n] = precision_matrices[n] / len(train) #+ self.args.eta

                p.data = param_w_weight
        # print(count)
        # sys.exit()

        return precision_matrices
    
