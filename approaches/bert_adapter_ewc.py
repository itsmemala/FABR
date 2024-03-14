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
import pickle
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


class Appr(ApprBase):


    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('BERT ADAPTER EWC NCL')

        return

    def train(self,t,train,valid,args,num_train_steps,save_path,train_data,valid_data):

        mcl_model=utils.get_model(self.model) # Main model before current task training
        
        global_step = 0
        self.model.to(self.device)

        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        if self.args.remove_wd:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer], 'weight_decay': 0.0}
                ]
            optimizer_param_keys = [n for n, p in param_optimizer]
        else:
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            optimizer_param_keys = [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)] +\
                                    [n for n, p in param_optimizer if any(nd in n for nd in no_decay)]
        if self.args.remove_lr_schedule:
            t_total = -1
            warmup = -1
        else:
            t_total = num_train_steps
            warmup = self.args.warmup_proportion
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=warmup,
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
        # best_f1=0
        best_model=utils.get_model(self.model)
        patience=self.args.lr_patience

        # Loop epochs
        epoch_wise_updates = np.array([])
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step,step_wise_updates=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step,class_counts,optimizer_param_keys)
            epoch_wise_updates = np.concatenate((epoch_wise_updates,step_wise_updates), axis=0)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            print('time: ',float((clock1-clock0)*10*25))
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, f1_avg={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_f1_macro),end='')

            valid_loss,valid_acc,valid_f1_macro=self.eval_validation(t,valid,class_counts=valid_class_counts)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_f1_macro),end='')
            
            # Adapt lr
            if best_loss-valid_loss > args.valid_loss_es:
            # if valid_f1_macro-best_f1 > self.args.valid_f1_es:
                patience=self.args.lr_patience
                # print(' *',end='')
            else:
                patience-=1
            if valid_loss<best_loss:
            # if valid_f1_macro>best_f1:
                best_loss=valid_loss
                # best_f1=valid_f1_macro
                best_model=utils.get_model(self.model)
                print(' *',end='')
            if patience<=0:
                break

            print()
            # break
        np.savetxt(save_path+str(args.note)+'_seed'+str(args.seed)+'_task'+str(t)+'_step_wise_updates.txt',epoch_wise_updates,'%.2f',delimiter='\t')
        
        # Restore best
        utils.set_model_(self.model,best_model)
        
        # Save model
        # torch.save(self.model.state_dict(), save_path+str(args.note)+'_seed'+str(args.seed)+'_model'+str(t))

        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()

        # if 'dil' in self.args.scenario:
            # self.fisher=utils.fisher_matrix_diag_bert_dil(t,train_data,self.device,self.model,self.criterion)
        # elif 'til' in self.args.scenario or 'cil' in self.args.scenario:
            # self.fisher=utils.fisher_matrix_diag_bert(t,train_data,self.device,self.model,self.criterion,scenario=self.args.scenario,imp=self.args.imp)
        self.fisher=utils.fisher_matrix_diag_bert(t,train_data,self.device,self.model,self.criterion,scenario=args.scenario,imp=self.args.imp,adjust_final=self.args.adjust_final,imp_layer_norm=self.args.imp_layer_norm)

        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                if self.args.fisher_combine=='avg': #default
                    self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option
                    #self.fisher[n]=0.5*(self.fisher[n]+fisher_old[n])
                elif self.args.fisher_combine=='max':
                    self.fisher[n]=torch.maximum(self.fisher[n],fisher_old[n])
                elif self.args.fisher_combine=='sum':
                    self.fisher[n]=torch.add(self.fisher[n],fisher_old[n])
                    
        if self.args.use_lamb_max==True:
            # Set EWC lambda for subsequent task
            vals = np.array([])
            for n in self.fisher.keys():
                vals = np.append(vals,self.fisher[n].detach().cpu().flatten().numpy())
            self.lamb = 1/(self.args.learning_rate*np.max(vals))
        elif self.args.use_ind_lamb_max==True:
            # Set EWC lambda for subsequent task
            for n in self.fisher.keys():
                self.lamb[n] = (1/(self.args.learning_rate*self.fisher[n]))/self.args.lamb_div
                self.lamb[n] = torch.clip(self.lamb[n],min=torch.finfo(self.lamb[n].dtype).min,max=torch.finfo(self.lamb[n].dtype).max)
        elif self.args.custom_lamb is not None:
            # Set EWC lambda for subsequent task
            self.lamb = self.args.custom_lamb[t+1]
        
        if t>0:
            wd_old = 0
            wd_old_magn = {}
            for n,param in self.model.named_parameters():
                if 'output.adapter' in n or 'output.LayerNorm' in n or (self.args.modify_fisher_last==True and 'last' in n):
                    wd_old += torch.sum((param.detach() - mcl_model[n].detach())**2).item()
                    # wd_old_magn[n] = math.sqrt(torch.sum((param.detach() - mcl_model[n].detach())**2).item())
                    wd_old_magn[n] = (param.detach() - mcl_model[n].detach())**2
            wd_old = math.sqrt(wd_old)
            np.savetxt(save_path+str(args.note)+'_seed'+str(args.seed)+'_task'+str(t)+'wd.txt',np.array([0,wd_old]),'%.4f',delimiter='\t')
            if self.args.save_wd_old_magn:
                with open(save_path+str(args.note)+'_seed'+str(args.seed)+'_task'+str(t)+'_wd_old_magn.pkl', 'wb') as fp:
                    pickle.dump(wd_old_magn, fp)

        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step,class_counts,optimizer_param_keys):
        self.num_labels = self.taskcla[t][1]
        self.model.train()
        step_wise_updates = []
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch

            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            # Forward
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]
            elif 'cil' in self.args.scenario:
                output=output_dict['y']
            loss=self.criterion(t,output,targets,class_counts=class_counts,phase='mcl')

            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()

            if self.args.remove_lr_schedule:
                lr_this_step = self.args.learning_rate
            else:  
                lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            _,updates = optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            if t>0:
                orthogonal_upds = 0
                num_params = 0 
                for k,param_old in self.model_old.named_parameters():
                    try:
                        param_upd = updates[optimizer_param_keys.index(k)]
                        param_old = param_old.detach().cpu().numpy()
                        assert param_old.shape == param_upd.shape
                        # print(k, param_upd.shape)
                        unit_x = param_upd / np.linalg.norm(param_upd)
                        unit_y = param_old / np.linalg.norm(param_old)
                        angle_rad = np.arccos(np.dot(unit_x, unit_y))
                        angle_deg = np.degrees(angle_rad)
                        # print(k, angle_deg, np.linalg.norm(param_upd), np.linalg.norm(param_old))
                        num_params += 1
                        if angle_deg > 89 and angle_deg < 91:
                            orthogonal_upds += 1
                    except ValueError:
                        continue # Skip parameters that are not being optimized
                step_wise_updates.append(orthogonal_upds*100/num_params)
            # break
                    

        return global_step,np.array(step_wise_updates)

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
                input_ids, segment_ids, input_mask, targets, _= batch
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
                # loss=self.criterion(t,output,targets)
                if 'cil' in self.args.scenario and self.args.use_rbs:
                    loss=self.ce(t,output,targets)
                else:
                    loss=self.criterion(t,output,targets,phase='mcl')

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
    
    def eval_validation(self,t,data,test=None,trained_task=None,class_counts=None):
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
                input_ids, segment_ids, input_mask, targets, _= batch
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
                
                # # loss=self.criterion(t,output,targets)
                # loss=self.ce(output,targets)
                
                # if 'cil' in self.args.scenario and self.args.use_rbs:
                    # loss=self.ce(t,output,targets,class_counts)
                # else:
                    # loss=self.ce(output,targets)
                loss=self.criterion(t,output,targets,class_counts=class_counts,phase='mcl')

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

