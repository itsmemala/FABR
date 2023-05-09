import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm, trange
import random
import json
import os
import math

class Appr(object):

    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x


    def __init__(self,model,logger,taskcla,args=None):
        self.model=model
        self.initial_model = deepcopy(model)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # Commented block as not sure what this does - args.__data__ is not defined in config.py on the repo
        # # print('before:',args.output_dir)
        # args.output_dir = args.output_dir.replace(
            # '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
        # # print('after:',args.output_dir)
        # os.makedirs(args.output_dir, exist_ok=True)
        # json.dump(args.__dict__, open(os.path.join(
            # args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        logger.info("device: {} n_gpu: {}".format(
            self.device, self.n_gpu))


        self.model=model

        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.args=args
        if args.experiment=='annomi' and args.use_cls_wgts==True:
            print('Using cls wgts')
            if args.convert_to_binary is None:
                class_weights = [0.41, 0.89, 0.16] #'change': 0, 'sustain': 1, 'neutral': 2
            elif args.convert_to_binary=='neutral_vs_other':
                class_weights = [0.28, 0.16] #'change/sustain': 0, 'neutral': 1
            class_weights = torch.FloatTensor(class_weights).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion=torch.nn.CrossEntropyLoss()

        print('DIL BERT MTL')

        return

    # def train(self,t,train,valid,num_train_steps,train_data,valid_data): # Commented as last 2 arguments aren't used
    def train(self,t,train,valid,args,num_train_steps,save_path):
        self.model=deepcopy(self.initial_model) # Restart model

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
        patience=self.args.lr_patience
        lr=self.args.learning_rate
        
        train_loss_save = []
        valid_loss_save = []
        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step)
            clock1=time.time()

            train_loss=self.eval_validation(t,train)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f} |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss),end='')
            train_loss_save.append(train_loss)

            valid_loss=self.eval_validation(t,valid)
            print(' Valid: loss={:.3f} |'.format(valid_loss),end='')
            valid_loss_save.append(valid_loss)
            
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                print(' *',end='')
            # if best_loss-valid_loss > args.valid_loss_es:
                # best_loss=valid_loss
                # best_model=utils.get_model(self.model)
                # patience=self.args.lr_patience
                # print(' *',end='')
            # else:
                # patience-=1
                # if patience<=0:
                    # break
                    # # lr/=self.args.lr_factor
                    # # print(' lr={:.1e}'.format(lr),end='')
                    # # if lr<self.args.lr_min:
                        # # print()
                        # # break
                    # # patience=self.args.lr_patience
                    # # optimizer=self._get_optimizer(lr,which_type)

            print()

        np.savetxt(save_path+args.experiment+'_'+args.approach+'_train_loss_'+str(t)+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',train_loss_save,'%.4f',delimiter='\t')
        np.savetxt(save_path+args.experiment+'_'+args.approach+'_valid_loss_'+str(t)+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',valid_loss_save,'%.4f',delimiter='\t')

        # Restore best
        utils.set_model_(self.model,best_model)

        # Save model
        # torch.save(self.model.state_dict(), save_path+str(args.note)+'_seed'+str(args.seed)+'_model'+str(t))

        return

    def train_epoch(self,_,data,iter_bar,optimizer,t_total,global_step):
        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, tasks= batch

            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            outputs = output_dict['y']

            # Forward
            loss=self.criterion_train(tasks,outputs,targets)

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

    def eval_validation(self,_,data):
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
                loss=self.criterion_train(tasks,outputs,targets)

                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_num+=real_b


        return total_loss/total_num

    def eval(self,t,data,test=None,trained_task=None):
        # This is used for the test. All tasks separately
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
                # Forward
                # output=outputs #shared head
                loss=self.criterion(output,targets)

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

            output_dict = self.model.forward(input_ids, segment_ids, input_mask)
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]
            # Forward
            # output=outputs #shared head
            # loss=self.criterion(output,targets)

            _,pred=output.max(1)
            hits=(pred==targets).float()
            target_list.append(targets)
            pred_list.append(pred)
            # Log
            # total_loss+=loss.data.cpu().numpy().item()*real_b
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
                # print('--------------------------')
                # print(input_ids.shape)
                # occ_output_b = self.model.forward(my_input_ids, my_segment_ids, my_input_mask)['y'][t]
                loop_size=4
                for j in range(math.ceil(my_input_ids.shape[0]/loop_size)):
                    occ_output_b = self.model.forward(my_input_ids[j*loop_size:j*loop_size+loop_size,:]
                                                    , my_segment_ids[j*loop_size:j*loop_size+loop_size,:]
                                                    , my_input_mask[j*loop_size:j*loop_size+loop_size,:])['y'][t]
                    occ_output_b = occ_output_b.detach().cpu()
                    if j==0:
                        occ_output = occ_output_b
                    else:
                        occ_output = torch.cat((occ_output,occ_output_b),axis=0)
                # occ_output = torch.nn.Softmax(dim=1)(occ_output)
                actual_output = self.model.forward(input_ids[i:i+1,:], segment_ids[i:i+1,:], input_mask[i:i+1,:])['y'][t]
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


    def criterion_train(self,tasks,outputs,targets):
        loss=0
        for t in np.unique(tasks.data.cpu().numpy()):
            t=int(t)
            # output = outputs  # shared head

            if 'dil' in self.args.scenario:
                output=outputs #always shared head
            elif 'til' in self.args.scenario:
                output = outputs[t]

            idx=(tasks==t).data.nonzero().view(-1)
            # print('Debugging:',output.shape,output[0])
            # print('Debugging:',targets.shape,targets[0])
            loss+=self.criterion(output[idx,:],targets[idx])*len(idx)
        return loss/targets.size(0)

    def f1_compute_fn(self,y_true, y_pred,average):
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred,average=average)
