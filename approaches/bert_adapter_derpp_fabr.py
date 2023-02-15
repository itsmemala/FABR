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
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils
# from apex import amp

from captum.attr import LayerIntegratedGradients

import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.append("./approaches/base/")
# from bert_adapter_base import Appr as ApprBase
from .bert_adapter_base import Appr as ApprBase
from .my_optimization import BertAdam


class Appr(ApprBase):


    def __init__(self,model,logger, taskcla=None,args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('DIL BERT Adapter NCL')

        return

    def train(self,t,train,valid,args,num_train_steps,save_path,train_data,valid_data):
    # def train(self,t,train,valid,num_train_steps,train_data,valid_data):
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


        best_loss=np.inf
        best_model=utils.get_model(self.model)

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            # iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX, ce loss=X.XXX, bce loss=X.XXX, bl loss=X.XXX, bfa loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step,args)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            # print('time: ',float((clock1-clock0)*10*25))

            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')

            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                print(' *',end='')

            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)

        # Save model
        torch.save(self.model.state_dict(), save_path+str(args.note)+'_seed'+str(args.seed)+'_model'+str(t))

        # add data to the buffer
        print('len(train): ',len(train_data))
        samples_per_task = int(len(train_data) * self.args.buffer_percent)
        print('samples_per_task: ',samples_per_task)

        loader = DataLoader(train_data, batch_size=samples_per_task)
        input_ids, segment_ids, input_mask, targets,_ = next(iter(loader))

        input_ids = input_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        targets = targets.to(self.device)


        output_dict = self.model.forward(input_ids, segment_ids, input_mask)
        if 'dil' in self.args.scenario:
            cur_task_output=output_dict['y']
        elif 'til' in self.args.scenario:
            outputs=output_dict['y']
            cur_task_output = outputs[t]

        if args.fa_method=='ig':
            self.model.train()
            integrated_gradients = LayerIntegratedGradients(self.model, self.model.bert.embeddings)
            # loop through inputs to avoid cuda memory err
            loop_size=4
            for i in range(math.ceil(input_ids.shape[0]/loop_size)):
                # print(i)
                attributions_ig_b = integrated_gradients.attribute(inputs=input_ids[i*loop_size:i*loop_size+loop_size,:]
                                                                    # Note: Attributions are not computed with respect to these additional arguments
                                                                    , additional_forward_args=(segment_ids[i*loop_size:i*loop_size+loop_size,:], input_mask[i*loop_size:i*loop_size+loop_size,:]
                                                                                              ,args.fa_method, t)
                                                                    , target=targets[i*loop_size:i*loop_size+loop_size], n_steps=10 # Attributions with respect to actual class
                                                                    # ,baselines=(baseline_embedding)
                                                                    )
                attributions_ig_b = attributions_ig_b.detach().cpu()
                # Get the max attribution across embeddings per token
                attributions_ig_b = torch.sum(attributions_ig_b, dim=2)
                if i==0:
                    attributions_ig = attributions_ig_b
                else:
                    attributions_ig = torch.cat((attributions_ig,attributions_ig_b),axis=0)
            # print('Input shape:',input_ids.shape)
            # print('IG attributions:',attributions_ig.shape)
            # print('Attributions:',attributions_ig[0,:])
            attributions = attributions_ig

        elif args.fa_method=='occ1':
            occ_mask = torch.ones((input_ids.shape[1],input_ids.shape[1])).to('cuda:0')
            for token in range(input_ids.shape[1]):
                occ_mask[token,token] = 0 # replace with padding token

            for i in range(len(input_ids)): # loop through each input in the new buffer data
                # print(i)
                temp_input_ids = input_ids[i:i+1,:] #.detach().clone().to('cuda:0') # using input_ids[:1,:] instead of input_ids[0] maintains the 2D shape of the tensor
                my_input_ids = (temp_input_ids*occ_mask).long()
                my_segment_ids = segment_ids[i:i+1,:].repeat(segment_ids.shape[1],1)
                my_input_mask = input_mask[i:i+1,:].repeat(input_mask.shape[1],1)
                if 'til' in self.args.scenario:
                    # occ_output = self.model.forward(my_input_ids, my_segment_ids, my_input_mask)['y'][t]
                    # occ_output = occ_output.detach().cpu()
                    # loop the forward to avoid cuda memory err
                    for j in range(math.ceil(my_input_ids.shape[0]/self.train_batch_size)):
                        start_idx = j*self.train_batch_size
                        end_idx = j*self.train_batch_size+self.train_batch_size
                        occ_output_b = self.model.forward(my_input_ids[start_idx:end_idx,:], my_segment_ids[start_idx:end_idx,:], my_input_mask[start_idx:end_idx,:])['y'][t]
                        occ_output_b = occ_output_b.detach().cpu()
                        if j==0:
                            occ_output = occ_output_b
                        else:
                            occ_output = torch.cat((occ_output,occ_output_b),0)
                # occ_output = torch.nn.Softmax(dim=1)(occ_output)
                actual_output = self.model.forward(input_ids[i:i+1,:], segment_ids[i:i+1,:], input_mask[i:i+1,:])['y'][t]
                actual_output = actual_output.detach().cpu()
                # actual_output = torch.nn.Softmax(dim=1)(actual_output)
                _,actual_pred = actual_output.max(1)
                _,occ_pred=occ_output.max(1)
                attributions_occ1_b = torch.subtract(actual_output,occ_output)[:,[actual_pred.item()]] # attributions towards the predicted class
                attributions_occ1_b = torch.transpose(attributions_occ1_b, 0, 1)
                attributions_occ1_b = attributions_occ1_b
                if i==0:
                    attributions_occ1 = attributions_occ1_b
                else:
                    attributions_occ1 = torch.cat((attributions_occ1,attributions_occ1_b), axis=0)
            attributions = attributions_occ1

        self.buffer.add_data(
            examples=input_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            labels=targets,
            task_labels=torch.ones(samples_per_task,dtype=torch.long).to(self.device) * (t),
            logits = cur_task_output.data,
            attributions=attributions
        )


        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step,args=None):
        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch

            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]


            loss_ce = self.ce(output,targets)


            if not self.buffer.is_empty():
                buf_inputs, buf_labels,buf_logits, buf_task_labels, buf_segment_ids,buf_input_mask,buf_attr_targets = self.buffer.get_data(
                    self.args.buffer_size)

                buf_task_inputs = buf_inputs.long().to('cuda')
                buf_task_segment = buf_segment_ids.long().to('cuda')
                buf_task_mask = buf_input_mask.long().to('cuda')
                buf_task_labels = buf_labels.long().to('cuda')
                buf_task_logits = buf_logits.to('cuda')
                buf_attr_targets = buf_attr_targets.to('cuda')

                output_dict = self.model.forward(buf_task_inputs, buf_task_segment, buf_task_mask)

                if 'dil' in self.args.scenario:
                    cur_task_output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    cur_task_output = outputs[t]

                loss_bce = self.args.beta * self.ce(cur_task_output, buf_task_labels)
                loss_bl = self.args.alpha * self.mse(cur_task_output, buf_task_logits)
                
                if args.fa_method=='ig':
                    integrated_gradients = LayerIntegratedGradients(self.model, self.model.bert.embeddings)
                    # loop through inputs to avoid cuda memory err
                    loop_size=2
                    for i in range(math.ceil(buf_task_inputs.shape[0]/loop_size)):
                        # print(i)
                        attributions_ig_b = integrated_gradients.attribute(inputs=buf_task_inputs[i*loop_size:i*loop_size+loop_size,:]
                                                                            # Note: Attributions are not computed with respect to these additional arguments
                                                                            , additional_forward_args=(buf_task_segment[i*loop_size:i*loop_size+loop_size,:], buf_task_mask[i*loop_size:i*loop_size+loop_size,:]
                                                                                                      ,args.fa_method, t)
                                                                            , target=buf_task_labels[i*loop_size:i*loop_size+loop_size], n_steps=10 # Attributions with respect to actual class
                                                                            # ,baselines=(baseline_embedding)
                                                                            )
                        # Get the max attribution across embeddings per token
                        attributions_ig_b = torch.sum(attributions_ig_b, dim=2)
                        if i==0:
                            attributions_ig = attributions_ig_b
                        else:
                            attributions_ig = torch.cat((attributions_ig,attributions_ig_b),axis=0)
                    # print('Input shape:',input_ids.shape)
                    # print('IG attributions:',attributions_ig.shape)
                    # print('Attributions:',attributions_ig[0,:])
                    attributions = attributions_ig
                
                elif args.fa_method=='occ1':
                    occ_mask = torch.ones((buf_task_inputs.shape[1],buf_task_inputs.shape[1])).to('cuda:0')
                    for token in range(buf_task_inputs.shape[1]):
                        occ_mask[token,token] = 0 # replace with padding token

                    for i in range(len(buf_task_inputs)): # loop through each input in the new buffer data
                        temp_buf_task_inputs = buf_task_inputs[i:i+1,:] #.detach().clone().to('cuda:0') # using buf_task_inputs[:1,:] instead of buf_task_inputs[0] maintains the 2D shape of the tensor
                        my_buf_task_inputs = (temp_buf_task_inputs*occ_mask).long()
                        my_buf_task_segment = buf_task_segment[i:i+1,:].repeat(buf_task_segment.shape[1],1)
                        my_buf_task_mask = buf_task_mask[i:i+1,:].repeat(buf_task_mask.shape[1],1)
                        if 'til' in self.args.scenario:
                            # occ_output = self.model.forward(my_buf_task_inputs, my_buf_task_segment, my_buf_task_mask)['y'][t]
                            # loop the forward to avoid cuda memory err
                            loop_size = 1
                            for j in range(math.ceil(my_buf_task_inputs.shape[0]/loop_size)):
                                print(j)
                                start_idx = j*loop_size
                                end_idx = j*loop_size+loop_size
                                occ_output_b = self.model.forward(my_buf_task_inputs[start_idx:end_idx,:], my_buf_task_segment[start_idx:end_idx,:], my_buf_task_mask[start_idx:end_idx,:])['y'][t]
                                occ_output_b = occ_output_b#.detach().cpu()
                                if j==0:
                                    occ_output = occ_output_b
                                else:
                                    occ_output = torch.cat((occ_output,occ_output_b),0)
                        # occ_output = torch.nn.Softmax(dim=1)(occ_output)
                        actual_output = self.model.forward(buf_task_inputs[i:i+1,:], buf_task_segment[i:i+1,:], buf_task_mask[i:i+1,:])['y'][t]
                        # actual_output = torch.nn.Softmax(dim=1)(actual_output)
                        _,actual_pred = actual_output.max(1)
                        _,occ_pred=occ_output.max(1)
                        attributions_occ1_b = torch.subtract(actual_output,occ_output)[:,[actual_pred.item()]] # attributions towards the predicted class
                        attributions_occ1_b = torch.transpose(attributions_occ1_b, 0, 1)
                        attributions_occ1_b = attributions_occ1_b #.detach().cpu()
                        if i==0:
                            attributions_occ1 = attributions_occ1_b
                        else:
                            attributions_occ1 = torch.cat((attributions_occ1,attributions_occ1_b), axis=0)
                    attributions = attributions_occ1
                
                loss_bfa = args.lfa_lambda*torch.square(torch.sum(torch.abs(buf_attr_targets-attributions)))
            else:
                loss_bce = torch.Tensor([0]).to('cuda:0')
                loss_bl = torch.Tensor([0]).to('cuda:0')
                loss_bfa = torch.Tensor([0]).to('cuda:0')
            
            loss = loss_ce + loss_bce + loss_bl + loss_bfa



            # iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            iter_bar.set_description('Train Iter (loss=%5.3f, ce loss=%5.3f, bce loss=%5.3f, bl loss=%5.3f, bfa loss=%5.3f)' % (loss.item(), loss_ce.item(), loss_bce.item(), loss_bl.item(), loss_bfa.item()))
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
        self.model.eval()
        target_list = []
        pred_list = []

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                output_dict = self.model.forward(input_ids, segment_ids, input_mask)
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]

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

