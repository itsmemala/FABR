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

sys.path.append("./approaches/base/")
# from bert_adapter_mask_base import Appr as ApprBase
from .bert_adapter_mask_base import Appr as ApprBase
from .my_optimization import BertAdam


class Appr(ApprBase):

    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('BERT ADAPTER CAPSULE MASK NCL')

        return

    # def train(self,t,train,valid,num_train_steps,train_data,valid_data): # Commented as the last 2 args are not used
    def train(self,t,train,valid,args,num_train_steps,save_path):
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
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train)
            clock2=time.time()
            self.logger.info('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))
            # print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
            #     1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')


            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid)
            self.logger.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc))
            # print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')

            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                self.logger.info(' *')
                # print(' *',end='')

            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)

        print('Training over.')

        # Activations mask
        # task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
        mask=self.model.mask(t,s=self.smax)
        for key,value in mask.items():
            mask[key]=torch.autograd.Variable(value.data.clone(),requires_grad=False)

        if t==0:
            self.mask_pre=mask
        else:
            for key,value in self.mask_pre.items():
                self.mask_pre[key]=torch.max(self.mask_pre[key],mask[key])

        # Weights mask
        self.mask_back={}
        for n,p in self.model.named_parameters():
            vals=self.model.get_view_for(n,p,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals

        print('Saving model..')

        # Save model
        torch.save(self.model.state_dict(), save_path+str(args.note)+'_seed'+str(args.seed)+'_model'+str(t))

        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step):
        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax

            output_dict = self.model.forward(t,input_ids, segment_ids, input_mask,targets,s=s)
            # Forward
            masks = output_dict['masks']
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]
            loss,_=self.hat_criterion_adapter(output,targets,masks)

            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back and p.grad is not None:
                        p.grad.data*=self.mask_back[n]
                    elif n in self.tsv_para and p.grad is not None:
                        p.grad.data*=self.model.get_view_for_tsv(n,t) #open for general

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if ('adapter_capsule_mask.e' in n or 'tsv_capsules.e' in n) and p.grad is not None: # we dont want etsv
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if 'adapter_capsule_mask.e' in n or 'tsv_capsules.e' in n:
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

            # break
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
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                output_dict = self.model.forward(t,input_ids, segment_ids, input_mask,targets,s=self.smax)
                masks = output_dict['masks']
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs=output_dict['y']
                    output = outputs[t]
                # Forward
                loss,_=self.hat_criterion_adapter(output,targets,masks)

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
        # This is used for the test. All tasks separately
        total_loss=0
        total_acc=0
        total_num=0
        # self.model.eval() #fixes params and randomness in the model # Do this only once after model is trained to ensure repeatable results when called for attribution calc
        target_list = []
        pred_list = []
        
        for step, batch in enumerate(data):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch
            real_b=input_ids.size(0)

            output_dict = self.model.forward(t,input_ids, segment_ids, input_mask,targets,s=self.smax)
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]

            _,pred=output.max(1)
            hits=(pred==targets).float()
            target_list.append(targets)
            pred_list.append(pred)
            # Log
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=real_b
            
            print('step:',step)
            occ_mask = torch.ones((input_ids.shape[1]-2,input_ids.shape[1])).to('cuda:0')
            for token in range(input_ids.shape[1]-2):
                occ_mask[token,token+1] = 0 # replace with padding token

            for i in range(len(input_ids)): # loop through each input in the batch
                temp_input_ids = input_ids[i:i+1,:].detach().clone().to('cuda:0') # using input_ids[:1,:] instead of input_ids[0] maintains the 2D shape of the tensor
                my_input_ids = (temp_input_ids*occ_mask).long()
                my_segment_ids = segment_ids[i:i+1,:].repeat(segment_ids.shape[1]-2,1)
                my_input_mask = input_mask[i:i+1,:].repeat(input_mask.shape[1]-2,1)
                my_targets = targets[i:i+1].repeat(len(targets)-2,1)
                # print('--------------------------')
                # print(input_ids.shape)
                # occ_output_b = self.model.forward(my_input_ids, my_segment_ids, my_input_mask)['y'][t]
                loop_size=4
                for j in range(math.ceil(my_input_ids.shape[0]/loop_size)):
                    occ_output_b = self.model.forward(t,my_input_ids[j*loop_size:j*loop_size+loop_size,:]
                                                    , my_segment_ids[j*loop_size:j*loop_size+loop_size,:]
                                                    , my_input_mask[j*loop_size:j*loop_size+loop_size,:]
                                                    ,my_targets[j*loop_size:j*loop_size+loop_size]
                                                    ,s=self.smax)['y'][t]
                    occ_output_b = occ_output_b.detach().cpu()
                    if j==0:
                        occ_output = occ_output_b
                    else:
                        occ_output = torch.cat((occ_output,occ_output_b),axis=0)
                # occ_output = torch.nn.Softmax(dim=1)(occ_output)
                actual_output = self.model.forward(t,input_ids[i:i+1,:], segment_ids[i:i+1,:], input_mask[i:i+1,:],targets[i:i+1],s=self.smax)['y'][t]
                actual_output = actual_output.detach().cpu()
                # actual_output = torch.nn.Softmax(dim=1)(actual_output)
                occ_output = torch.cat((actual_output,occ_output,actual_output), axis=0) # placeholder for CLS and SEP such that their attribution scores are 0
                _,actual_pred = actual_output.max(1)
                _,occ_pred=occ_output.max(1)
                # print(occ_output)
                # print(actual_output)
                attributions_occ1_b = torch.subtract(actual_output,occ_output)[:,[actual_pred.item()]] # attributions towards the predicted class
                attributions_occ1_b = torch.transpose(attributions_occ1_b, 0, 1)
                attributions_occ1_b = attributions_occ1_b.detach().cpu()
                
                if step==0 and i==0:
                    attributions_occ1 = attributions_occ1_b
                else:
                    attributions_occ1 = torch.cat((attributions_occ1,attributions_occ1_b), axis=0)
            attributions = attributions_occ1

        return torch.cat(target_list,0),torch.cat(pred_list,0),attributions

