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

        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)


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
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step,class_counts)
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

        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step,class_counts):
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
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
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
                loss+=self.ce(output[idx,:],targets[idx])*len(idx)                
            
            # loss2+=self.ce2(output[idx,:],targets[idx])*len(idx)
        # try:
            # assert loss.item()==loss2.item()
        # except AssertionError:
            # print(loss.item(),loss2.item()) #TODO: Check why there is variation after the 4th decimal
        
        return loss/targets.size(0)
    
    def get_attributions(self,t,data,input_tokens=None):
        target_list = []
        pred_list = []


        # with torch.no_grad():
            # self.model.eval()

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

            _,pred=output.max(1)
            hits=(pred==targets).float()

            target_list.append(targets)
            pred_list.append(pred)


            # Calculate attributions
            integrated_gradients = LayerIntegratedGradients(self.model, self.model.bert.embeddings)
            # loop through inputs to avoid cuda memory err
            loop_size=4
            for i in range(math.ceil(input_ids.shape[0]/loop_size)):
                # print(i)
                attributions_ig_b = integrated_gradients.attribute(inputs=input_ids[i*loop_size:i*loop_size+loop_size,:]
                                                                    # Note: Attributions are not computed with respect to these additional arguments
                                                                    , additional_forward_args=(segment_ids[i*loop_size:i*loop_size+loop_size,:], input_mask[i*loop_size:i*loop_size+loop_size,:]
                                                                                              ,self.args.fa_method, t)
                                                                    , target=pred[i*loop_size:i*loop_size+loop_size], n_steps=10 # Attributions with respect to predicted class
                                                                    # ,baselines=(baseline_embedding)
                                                                    )
                attributions_ig_b = attributions_ig_b.detach().cpu()
                # Get the max attribution across embeddings per token
                attributions_ig_b = torch.sum(attributions_ig_b, dim=2)
                if i==0 and step==0:
                    attributions_ig = attributions_ig_b
                else:
                    attributions_ig = torch.cat((attributions_ig,attributions_ig_b),axis=0)
            # print('Input shape:',input_ids.shape)
            # print('IG attributions:',attributions_ig.shape)
            # print('Attributions:',attributions_ig[0,:])
            attributions = attributions_ig
            # optimizer.zero_grad()


        return torch.cat(target_list,0),torch.cat(pred_list,0),attributions
