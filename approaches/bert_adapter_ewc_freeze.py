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
import pickle
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

class Appr(ApprBase):


    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('BERT ADAPTER LA-EWC')

        return

    def train(self,t,train,valid,args,num_train_steps,save_path,train_data,valid_data):

        if t>0:
            train_phases = ['fo','mcl']
        elif t==0 and self.args.regularize_t0==False:
            train_phases = ['mcl']
        elif t==0 and self.args.regularize_t0==True:
            train_phases = ['fo','mcl']
        if self.training_multi:
            train_phases = ['fo']
        
        for phase in train_phases:
            if phase=='fo':
                self.mcl_model=utils.get_model(self.model) # Save the main model before commencing fisher overlap check
            
            if t>0:
                torch.manual_seed(args.seed) # Ensure same shuffling order of dataloader and other random behaviour between fo and mcl phases
        
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
        
        
            train_loss_save,train_acc_save,train_f1_macro_save = [],[],[]
            valid_loss_save,valid_acc_save,valid_f1_macro_save = [],[],[]
            if phase == 'fo':
                epochs = self.args.la_num_train_epochs
            else:
                epochs = self.args.num_train_epochs
            
            # Loop epochs
            for e in range(int(epochs)):
                # if phase=='fo' and e==0 and t==3:
                    # # Fisher weights
                    # lastart_fisher,grad_dir_lastart=utils.fisher_matrix_diag_bert(t,train_data,self.device,self.model,self.criterion,scenario=args.scenario,imp=self.args.imp,adjust_final=self.args.adjust_final,imp_layer_norm=self.args.imp_layer_norm,get_grad_dir=True)
                    # # Save
                    # if self.args.save_metadata=='all':
                        # # Attributions
                        # targets, predictions, attributions = self.get_attributions(t,train)
                        # np.savez_compressed(save_path+str(args.note)+'_seed'+str(args.seed)+'_attributions_model'+str(t)+'task'+str(t)+'_lastart'
                                        # ,targets=targets.cpu()
                                        # ,predictions=predictions.cpu()
                                        # ,attributions=attributions.cpu()
                                        # )
                        # # Fisher weights
                        # with open(save_path+str(args.note)+'_seed'+str(args.seed)+'_lastart_fisher_task'+str(t)+'.pkl', 'wb') as fp:
                            # pickle.dump(lastart_fisher, fp)
                        # with open(save_path+str(args.note)+'_seed'+str(args.seed)+'_lastart_graddir_task'+str(t)+'.pkl', 'wb') as fp:
                            # pickle.dump(grad_dir_lastart, fp)
            
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                global_step,step_wise_updates=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step,class_counts=class_counts,phase=phase,optimizer_param_keys=optimizer_param_keys)
                clock1=time.time()

                train_loss,train_acc,train_f1_macro=self.eval(t,train,phase=phase)
                clock2=time.time()
                print('time: ',float((clock1-clock0)*10*25))
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, f1_avg={:5.1f}% |'.format(e+1,
                    1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_f1_macro),end='')
                train_loss_save.append(train_loss)
                train_acc_save.append(train_acc)
                train_f1_macro_save.append(train_f1_macro)

                valid_loss,valid_acc,valid_f1_macro=self.eval_validation(t,valid,class_counts=valid_class_counts,phase=phase)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_f1_macro),end='')
                valid_loss_save.append(valid_loss)
                valid_acc_save.append(valid_acc)
                valid_f1_macro_save.append(valid_f1_macro)
                
                # Adapt lr
                if best_loss-valid_loss > args.valid_loss_es:
                # if valid_f1_macro-best_f1 > self.args.valid_f1_es:
                    patience=self.args.lr_patience
                    # print(' *',end='')
                else:
                    patience-=1
                    # if patience<=0:
                        # break
                        # lr/=self.lr_factor
                        # print(' lr={:.1e}'.format(lr),end='')
                        # if lr<self.lr_min:
                            # print()
                            # break
                        # patience=self.args.lr_patience
                        # self.optimizer=self._get_optimizer(lr,which_type)
                if valid_loss<best_loss:
                # if valid_f1_macro>best_f1:
                    best_loss=valid_loss
                    # best_f1=valid_f1_macro
                    best_model=utils.get_model(self.model)
                    print(' *',end='')
                if patience<=0:
                    break

                print()

            try:
                best_index = valid_loss_save.index(best_loss)
                # best_index = valid_f1_macro_save.index(best_f1)
            except ValueError:
                best_index = -1
            np.savetxt(save_path+args.experiment+'_'+args.approach+'_'+phase+'_train_loss_'+str(t)+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',train_loss_save,'%.4f',delimiter='\t')
            np.savetxt(save_path+args.experiment+'_'+args.approach+'_'+phase+'_train_acc_'+str(t)+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',train_acc_save,'%.4f',delimiter='\t')
            np.savetxt(save_path+args.experiment+'_'+args.approach+'_'+phase+'_train_f1_macro_'+str(t)+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',train_f1_macro_save,'%.4f',delimiter='\t')    
            np.savetxt(save_path+args.experiment+'_'+args.approach+'_'+phase+'_valid_loss_'+str(t)+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',valid_loss_save,'%.4f',delimiter='\t')
            np.savetxt(save_path+args.experiment+'_'+args.approach+'_'+phase+'_valid_acc_'+str(t)+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',valid_acc_save,'%.4f',delimiter='\t')
            np.savetxt(save_path+args.experiment+'_'+args.approach+'_'+phase+'_valid_f1_macro_'+str(t)+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',valid_f1_macro_save,'%.4f',delimiter='\t')

            # Restore best
            utils.set_model_(self.model,best_model)
            
            # if self.args.save_metadata=='all'and phase=='fo' and t==3:
                # # Attributions
                # targets, predictions, attributions = self.get_attributions(t,train)
                # np.savez_compressed(save_path+str(args.note)+'_seed'+str(args.seed)+'_attributions_model'+str(t)+'task'+str(t)+'_laend'
                                # ,targets=targets.cpu()
                                # ,predictions=predictions.cpu()
                                # ,attributions=attributions.cpu()
                                # )
            
            # Save model
            # torch.save(self.model.state_dict(), save_path+str(args.note)+'_seed'+str(args.seed)+'_model'+str(t))

            if phase=='mcl':
                if t>0:
                    frozen_paramcount = 0
                    for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                        if torch.sum(param_old-param)==0:
                            frozen_paramcount+=1
                    print('Frozen paramcount:',frozen_paramcount)
                # Update old
                self.model_old=deepcopy(self.model)
                self.model_old.eval()
                utils.freeze_model(self.model_old) # Freeze the weights

            # Fisher ops
            if t>0 and phase=='fo':
                fisher_old={}
                for n,_ in self.model.named_parameters():
                    fisher_old[n]=self.fisher[n].clone()

            if self.training_multi:
                pass
            else:
                self.fisher,grad_dir_laend=utils.fisher_matrix_diag_bert(t,train_data,self.device,self.model,self.criterion,scenario=args.scenario,imp=self.args.imp,adjust_final=self.args.adjust_final,imp_layer_norm=self.args.imp_layer_norm,get_grad_dir=True)
            if t==0:
                self.fisher_for_loss = self.fisher
            # if  self.args.save_metadata=='all'and phase=='fo' and t==3:
                # with open(save_path+str(args.note)+'_seed'+str(args.seed)+'_laend_fisher_task'+str(t)+'.pkl', 'wb') as fp:
                    # pickle.dump(self.fisher, fp)
                # with open(save_path+str(args.note)+'_seed'+str(args.seed)+'_laend_graddir_task'+str(t)+'.pkl', 'wb') as fp:
                    # pickle.dump(grad_dir_laend, fp)
            

            if phase=='fo':
                # Freeze non-overlapping params
                # if t==3:
                    # self.fisher=utils.modified_fisher(self.fisher,fisher_old
                    # ,train_f1_macro_save,best_index
                    # ,self.model,self.model_old
                    # ,self.args.elasticity_down,self.args.elasticity_up
                    # ,self.args.freeze_cutoff
                    # ,self.args.learning_rate,self.args.lamb
                    # ,grad_dir_lastart,grad_dir_laend,lastart_fisher
                    # ,save_path+str(args.note)+'_seed'+str(args.seed)+'model_'+str(t))
                # else:
                self.fisher_for_loss=utils.modified_fisher(self.fisher,fisher_old
                ,train_f1_macro_save,best_index
                ,self.model,self.model_old
                ,self.args.elasticity_down,self.args.elasticity_up
                ,self.args.freeze_cutoff
                ,self.args.learning_rate,self.lamb,self.args.use_ind_lamb_max
                ,adapt_type=self.args.adapt_type
                ,ktcf_wgt=self.args.ktcf_wgt
                ,ktcf_wgt_use_arel=self.args.ktcf_wgt_use_arel
                ,frel_cut=self.args.frel_cut, frel_cut_type=self.args.frel_cut_type
                ,modify_fisher_last=self.args.modify_fisher_last
                ,save_alpharel=self.args.save_alpharel
                ,save_path=save_path+str(args.note)+'_seed'+str(args.seed)+'model_'+str(t))

            if t>0 and phase=='mcl':
                # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
                for n,_ in self.model.named_parameters():
                    if self.args.fisher_combine=='avg': #default
                        self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option
                        #self.fisher[n]=0.5*(self.fisher[n]+fisher_old[n])
                    elif self.args.fisher_combine=='max':
                        self.fisher[n]=torch.maximum(self.fisher[n],fisher_old[n])
                # with open(save_path+str(args.note)+'_seed'+str(args.seed)+'_fisher_task'+str(t)+'.pkl', 'wb') as fp:
                    # pickle.dump(self.fisher, fp)

            if phase=='mcl' and self.args.use_lamb_max==True:
                # Set EWC lambda for subsequent task
                vals = np.array([])
                for n in self.fisher.keys():
                    vals = np.append(vals,self.fisher[n].detach().cpu().flatten().numpy())
                self.lamb = 1/(self.args.learning_rate*np.max(vals))
            elif phase=='mcl' and self.args.use_ind_lamb_max==True:
                # Set EWC lambda for subsequent task
                for n in self.fisher.keys():
                    self.lamb[n] = (1/(self.args.learning_rate*self.fisher[n]))/self.args.lamb_div
                    self.lamb[n] = torch.clip(self.lamb[n],min=torch.finfo(self.lamb[n].dtype).min,max=torch.finfo(self.lamb[n].dtype).max)
            elif phase=='mcl' and self.args.custom_lamb is not None:
                # Set EWC lambda for subsequent task
                self.lamb = self.args.custom_lamb[t+1] if t+1<=self.args.break_after_task else 0

            if phase=='fo':
                self.la_model=utils.get_model(self.model)
                utils.set_model_(self.model,self.mcl_model) # Reset to main model after fisher overlap check
            
            if phase=='mcl' and t>0:
                wd_aux = 0
                wd_old = 0
                wd_old_magn = {}
                for n,param in self.model.named_parameters():
                    if 'output.adapter' in n or 'output.LayerNorm' in n or (self.args.modify_fisher_last==True and 'last' in n):
                        wd_aux += torch.sum((param.detach() - self.la_model[n].detach())**2).item()
                        wd_old += torch.sum((param.detach() - self.mcl_model[n].detach())**2).item()
                        # wd_old_magn[n] = math.sqrt(torch.sum((param.detach() - self.mcl_model[n].detach())**2).item())
                        wd_old_magn[n] = (param.detach() - self.mcl_model[n].detach())**2
                wd_aux = math.sqrt(wd_aux)
                wd_old = math.sqrt(wd_old)
                np.savetxt(save_path+str(args.note)+'_seed'+str(args.seed)+'_task'+str(t)+'wd.txt',np.array([wd_aux,wd_old]),'%.4f',delimiter='\t')
                if self.args.save_wd_old_magn:
                    with open(save_path+str(args.note)+'_seed'+str(args.seed)+'_task'+str(t)+'_wd_old_magn.pkl', 'wb') as fp:
                        pickle.dump(wd_old_magn, fp)
                
        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step,class_counts,phase=None,optimizer_param_keys=None):
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
            loss=self.criterion(t,output,targets,class_counts=class_counts,phase=phase)

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

        return global_step,np.array(step_wise_updates)

    def eval(self,t,data,test=None,trained_task=None,phase=None):
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
                loss=self.criterion(t,output,targets,phase=phase)

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
    
    def eval_validation(self,t,data,test=None,trained_task=None,class_counts=None,phase=None):
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
                loss=self.criterion(t,output,targets,class_counts=class_counts,phase=phase)

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
            loss=self.criterion(t,output,targets)

            _,pred=output.max(1)
            hits=(pred==targets).float()

            target_list.append(targets)
            pred_list.append(pred)


            # Calculate attributions
            integrated_gradients = LayerIntegratedGradients(self.model, self.model.bert.embeddings)
            # loop through inputs to avoid cuda memory err
            if t==2:
                loop_size=6
            else:
                loop_size=3
            for i in range(math.ceil(input_ids.shape[0]/loop_size)):
                # print(i)
                attributions_ig_b = integrated_gradients.attribute(inputs=input_ids[i*loop_size:i*loop_size+loop_size,:]
                                                                    # Note: Attributions are not computed with respect to these additional arguments
                                                                    , additional_forward_args=(segment_ids[i*loop_size:i*loop_size+loop_size,:], input_mask[i*loop_size:i*loop_size+loop_size,:]
                                                                                              ,self.args.fa_method, t)
                                                                    , target=targets[i*loop_size:i*loop_size+loop_size], n_steps=10 # Attributions with respect to actual class
                                                                    # ,baselines=(baseline_embedding)
                                                                    )
                attributions_ig_b = attributions_ig_b.detach().cpu()
                # Get the max attribution across embeddings per token
                # attributions_ig_b = torch.sum(attributions_ig_b, dim=2)
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
    
    def eval_temp_model(self,t,data,test=None,trained_task=None,class_counts=None,phase=None,use_model=None):
        total_loss=0
        total_acc=0
        total_num=0
        target_list = []
        pred_list = []


        with torch.no_grad():
            use_model.eval()

            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                output_dict = use_model.forward(input_ids, segment_ids, input_mask)
                # Forward
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]
                elif 'cil' in self.args.scenario:
                    output=output_dict['y']
                
                if 'cil' in self.args.scenario and self.args.use_rbs:
                    loss=self.ce(t,output,targets,class_counts)
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

        return total_loss/total_num,total_acc/total_num,f1
    
    
    def plot_loss_along_interpolation_line(self,t,valid_dataloader,valid_dataloader_past,test_dataloader,test_dataloader_past,fig_path):
        # Implementation based on: https://github.com/kim-sanghwan/ANCL/blob/main/src/mean_acc_landscape_analysis.py
        
        # for layer in range(12):
            # layer_vector_w1, layer_vector_w2, layer_vector_w3 = [], [], []
            # for n,param in self.model_old.named_parameters():
                # if str(layer) in n:
                    # layer_vector_w1.append(param.detach().cpu().flatten())
                    # layer_vector_w2.append(self.la_model[n].detach().cpu().flatten())
                    # layer_vector_w3.append(self.mcl_model[n].detach().cpu().flatten())
            # layer_vector_w1, layer_vector_w2, layer_vector_w3 = torch.cat(layer_vector_w1,axis=0), torch.cat(layer_vector_w2,axis=0), torch.cat(layer_vector_w3,axis=0)
            # assert len(layer_vector_w1.shape)==1 # Check that it is flattened
            # u = layer_vector_w2 - layer_vector_w1
            # v = layer_vector_w3 - layer_vector_w1
            # u_norm = torch.sqrt(torch.sum(u**2))
            # v = v - (torch.dot(u, v)/(u_norm**2))*u
        
        x_diff, temp_diff = 0, 0
        x_param_list, temp_param_list = {}, {}
        # Calculate weight vector model2-model1 and set it as axis x direction
        # Calculate weight vector model3-model1 and set is as temp direction
        for n,param in self.model_old.named_parameters():
            x_param_list[n] = self.plot_la_models[0][n].detach().cpu() - param.detach().cpu()
            x_diff += torch.sum(x_param_list[n]**2).item()
            temp_param_list[n] = self.multi_model[n].detach().cpu() - param.detach().cpu()
            temp_diff += torch.sum(temp_param_list[n]**2).item()
        
        # Calculate y axis given x and temp vector.  
        dot_product = 0
        for n,param in self.model_old.named_parameters():
            dot_product += torch.sum(temp_param_list[n] * x_param_list[n]).item()
        y_diff = 0
        x_pos = 0
        y_param_list = {}
        for n,param in self.model_old.named_parameters(): 
            y_param_list[n] = temp_param_list[n] - (dot_product/x_diff)* x_param_list[n]
            y_diff += torch.sum(y_param_list[n]**2).item()
            x_pos += torch.sum(((dot_product/x_diff)* x_param_list[n])**2).item()
        
        # Sanity check to see x and y axis is valid
        should_zero = 0
        for n,param in self.model_old.named_parameters(): 
            should_zero += torch.sum(x_param_list[n] * y_param_list[n]).item() 
        # print(f"should_zero {should_zero}")
        # try:
            # assert x_pos <= x_diff # This is not needed since the la model is not necessarily always farther than the mcl/multi model
        # except AssertionError:
            # print(x_diff,x_pos)
        print(x_diff,x_pos)
        
        x_diff = math.sqrt(x_diff)
        temp_diff = math.sqrt(temp_diff)
        x_pos = math.sqrt(x_pos)
        y_diff = math.sqrt(y_diff)
        
        # Calculate co-ordinates for all the LA and MCL model variants
        LA_VARIANT_x_pos_list, LA_VARIANT_y_pos_list, LA_VARIANT_info_list, plot_la_models_keys = [], [], [], []
        for la_idx,LA_VARIANT_model in self.plot_la_models.items():
            if la_idx==0: continue # This is already used to calculate x_diff so we skip
            plot_la_models_keys.append(la_idx)
            LA_VARIANT_xdot_product = 0
            LA_VARIANT_ydot_product = 0
            LA_VARIANT_param_list = {}
            for n,param in self.model_old.named_parameters(): 
                LA_VARIANT_param_list[n] = LA_VARIANT_model[n].detach().cpu() - param.detach().cpu()
                LA_VARIANT_xdot_product += torch.sum(LA_VARIANT_param_list[n] * x_param_list[n]).item()
                LA_VARIANT_ydot_product += torch.sum(LA_VARIANT_param_list[n] * y_param_list[n]).item()     

            LA_VARIANT_x_pos = 0
            LA_VARIANT_y_pos = 0
            LA_VARIANT_left_param_diff = 0
            LA_VARIANT_left_param_list = {}
            for n,param in self.model_old.named_parameters(): 
                LA_VARIANT_x_pos += torch.sum(((LA_VARIANT_xdot_product/x_diff)* x_param_list[n])**2).item()
                LA_VARIANT_y_pos += torch.sum(((LA_VARIANT_ydot_product/y_diff)* y_param_list[n])**2).item()

                LA_VARIANT_left_param_list[n] = LA_VARIANT_param_list[n] - (LA_VARIANT_xdot_product/x_diff)* x_param_list[n] \
                                                - (LA_VARIANT_ydot_product/y_diff)* y_param_list[n]
                LA_VARIANT_left_param_diff += torch.sum(LA_VARIANT_left_param_list[n]**2).item()
            LA_VARIANT_x_pos = math.sqrt(LA_VARIANT_x_pos)
            LA_VARIANT_y_pos = math.sqrt(LA_VARIANT_y_pos)
            LA_VARIANT_left_param_diff = math.sqrt(LA_VARIANT_left_param_diff)
            LA_VARIANT_info_list.append((LA_VARIANT_x_pos, LA_VARIANT_y_pos, LA_VARIANT_left_param_diff))
            LA_VARIANT_x_pos_list.append(LA_VARIANT_x_pos)
            LA_VARIANT_y_pos_list.append(LA_VARIANT_y_pos)
        MCL_VARIANT_x_pos_list, MCL_VARIANT_y_pos_list, MCL_VARIANT_info_list, plot_mcl_models_keys = [], [], [], []
        for mcl_idx,MCL_VARIANT_model in self.plot_mcl_models.items():
            plot_mcl_models_keys.append(mcl_idx)
            MCL_VARIANT_xdot_product = 0
            MCL_VARIANT_ydot_product = 0
            MCL_VARIANT_param_list = {}
            for n,param in self.model_old.named_parameters(): 
                MCL_VARIANT_param_list[n] = MCL_VARIANT_model[n].detach().cpu() - param.detach().cpu()
                MCL_VARIANT_xdot_product += torch.sum(MCL_VARIANT_param_list[n] * x_param_list[n]).item()
                MCL_VARIANT_ydot_product += torch.sum(MCL_VARIANT_param_list[n] * y_param_list[n]).item()     

            MCL_VARIANT_x_pos = 0
            MCL_VARIANT_y_pos = 0
            MCL_VARIANT_left_param_diff = 0
            MCL_VARIANT_left_param_list = {}
            for n,param in self.model_old.named_parameters(): 
                MCL_VARIANT_x_pos += torch.sum(((MCL_VARIANT_xdot_product/x_diff)* x_param_list[n])**2).item()
                MCL_VARIANT_y_pos += torch.sum(((MCL_VARIANT_ydot_product/y_diff)* y_param_list[n])**2).item()

                MCL_VARIANT_left_param_list[n] = MCL_VARIANT_param_list[n] - (MCL_VARIANT_xdot_product/x_diff)* x_param_list[n] \
                                                - (MCL_VARIANT_ydot_product/y_diff)* y_param_list[n]
                MCL_VARIANT_left_param_diff += torch.sum(MCL_VARIANT_left_param_list[n]**2).item()
            MCL_VARIANT_x_pos = math.sqrt(MCL_VARIANT_x_pos)
            MCL_VARIANT_y_pos = math.sqrt(MCL_VARIANT_y_pos)
            MCL_VARIANT_left_param_diff = math.sqrt(MCL_VARIANT_left_param_diff)
            MCL_VARIANT_info_list.append((MCL_VARIANT_x_pos, MCL_VARIANT_y_pos, MCL_VARIANT_left_param_diff))
            MCL_VARIANT_x_pos_list.append(MCL_VARIANT_x_pos)
            MCL_VARIANT_y_pos_list.append(MCL_VARIANT_y_pos)
        
        #Divide subspace with n*n points
        num_points = 50
        # x_max = x_diff if x_diff>x_pos else x_pos
        all_x_pos = LA_VARIANT_x_pos_list + MCL_VARIANT_x_pos_list
        all_x_pos.append(x_diff)
        all_x_pos.append(x_pos)
        all_y_pos = LA_VARIANT_y_pos_list + MCL_VARIANT_y_pos_list
        all_y_pos.append(y_diff)
        y_max = np.max(all_y_pos)
        xlist = np.linspace(-3/10*x_diff, 13/10*x_max, num_points)
        ylist = np.linspace(-3/10*y_diff, 13/10*y_max, num_points)
        X, Y = np.meshgrid(xlist, ylist)

        # Valid data
        Z1 = np.random.randn(num_points,num_points) #Task t loss landscape
        Z1_2 = np.random.randn(num_points,num_points) #Task t acc landscape
        Z1_3 = np.random.randn(num_points,num_points) #Task t f1 landscape

        Z2 = np.random.randn(num_points,num_points) #Task t-1 loss landscape
        Z2_2 = np.random.randn(num_points,num_points) #Task t-1 acc landscape
        Z2_3 = np.random.randn(num_points,num_points) #Task t-1 f1 landscape      

        # Test data
        Z3 = np.random.randn(num_points,num_points) #Task t loss landscape
        Z3_2 = np.random.randn(num_points,num_points) #Task t acc landscape
        Z3_3 = np.random.randn(num_points,num_points) #Task t f1 landscape

        Z4 = np.random.randn(num_points,num_points) #Task t-1 loss landscape
        Z4_2 = np.random.randn(num_points,num_points) #Task t-1 acc landscape
        Z4_3 = np.random.randn(num_points,num_points) #Task t-1 f1 landscape
        
        # Init temp model
        self.plot_model = deepcopy(self.model_old)
        model_old_dict = utils.get_model(self.model_old)
        
        # Calculate loss and accuracy at n*n points
        total_results = []
        for y_tick in tqdm(range(num_points)):
            x_results = []
            for x_tick in range(num_points):
                with torch.no_grad():
                   for n,param in self.plot_model.named_parameters(): 
                        param.copy_(model_old_dict[n].data.cpu() + xlist[x_tick]/x_diff* x_param_list[n] \
                            + ylist[y_tick]/y_diff* y_param_list[n])
        
                Z1[y_tick, x_tick], Z1_2[y_tick, x_tick], Z1_3[y_tick, x_tick] = self.eval_temp_model(t,valid_dataloader,use_model=self.plot_model)
                Z2[y_tick, x_tick], Z2_2[y_tick, x_tick], Z2_3[y_tick, x_tick] = self.eval_temp_model(t-1,valid_dataloader_past[-1],use_model=self.plot_model)
                Z3[y_tick, x_tick], Z3_2[y_tick, x_tick], Z3_3[y_tick, x_tick] = self.eval_temp_model(t,test_dataloader,use_model=self.plot_model)
                Z4[y_tick, x_tick], Z4_2[y_tick, x_tick], Z4_3[y_tick, x_tick] = self.eval_temp_model(t-1,test_dataloader_past[-1],use_model=self.plot_model)
        
        # denoise values to make contour smooth
        Z1 = gaussian_filter(Z1, 2)
        Z1_2 = gaussian_filter(Z1_2, 2)
        Z1_3 = gaussian_filter(Z1_3, 2)

        Z2 = gaussian_filter(Z2, 2)
        Z2_2 = gaussian_filter(Z2_2, 2)
        Z2_3 = gaussian_filter(Z2_3, 2)
        
        Z3 = gaussian_filter(Z3, 2)
        Z3_2 = gaussian_filter(Z3_2, 2)
        Z3_3 = gaussian_filter(Z3_3, 2)

        Z4 = gaussian_filter(Z4, 2)
        Z4_2 = gaussian_filter(Z4_2, 2)
        Z4_3 = gaussian_filter(Z4_3, 2)
        
        plot_names = ['cur_task_loss','cur_task_acc','cur_task_f1','past_task_loss','past_task_acc','past_task_f1',
                        'cur_task_testloss','cur_task_testacc','cur_task_testf1','past_task_testloss','past_task_testacc','past_task_testf1']
        # All plots
        for plot_name,vals in zip(plot_names,[Z1,Z1_2,Z1_3,Z2,Z2_2,Z2_3,Z3,Z3_2,Z3_3,Z4,Z4_2,Z4_3]):
            fig,ax=plt.subplots(figsize=(10, 7))
            cp = ax.contourf(X, Y, vals, cmap = 'gist_rainbow', alpha=0.6)
            fig.colorbar(cp) # Add a colorbar to a plot
            # First plot only the three models
            plt.plot(0, 0, 'o', c='black') 
            plt.plot(x_diff, 0, 'o', c='black') 
            plt.plot(x_pos, y_diff, 'o', c='black')
            plt.text(0, -y_diff/10,'$\u03F4^{{old}}$', fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
            plt.text(x_diff, -y_diff/10,'$\u03F4_{}^{{la}}$'.format(0), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
            plt.text(x_pos, y_diff-y_diff/10,'$\u03F4^{{multi}}$', fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
            # Now plot the LA and MCL variants
            for i, LA_VARIANT_info in enumerate(LA_VARIANT_info_list):
                plt.plot(LA_VARIANT_info[0], LA_VARIANT_info[1], marker='o', c='saddlebrown', markersize=8)
                plt.text(LA_VARIANT_info[0], LA_VARIANT_info[1]-y_diff/10,'$\u03F4_{}^{{la}}$'.format(plot_la_models_keys[i]), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
            for i, MCL_VARIANT_info in enumerate(MCL_VARIANT_info_list):
                plt.plot(MCL_VARIANT_info[0], MCL_VARIANT_info[1], marker='*', c='red', markersize=8)
                plt.text(MCL_VARIANT_info[0], MCL_VARIANT_info[1]-y_diff/10,'$\u03F4_{}^{{mcl}}$'.format(plot_mcl_models_keys[i]), fontsize=15.0, fontfamily= 'monospace', fontstyle = 'normal')
            
            plt.savefig(fig_path+'_'+plot_name+'.png')
            # break        
            
        return