import sys,time
import numpy as np
import torch
# from copy import deepcopy

import utils
from tqdm import tqdm, trange

rnn_weights = [
    'mcl.lstm.rnn.weight_ih_l0',
    'mcl.lstm.rnn.weight_hh_l0',
    'mcl.lstm.rnn.bias_ih_l0',
    'mcl.lstm.rnn.bias_hh_l0',
    'mcl.gru.rnn.weight_ih_l0',
    'mcl.gru.rnn.weight_hh_l0',
    'mcl.gru.rnn.bias_ih_l0',
    'mcl.gru.rnn.bias_hh_l0']

class Appr(object):
    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=3,clipgrad=10000,args=None,logger=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        self.model=model
        # self.initial_model=deepcopy(model)

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.smax = 400
        self.thres_cosh=50
        self.thres_emb=6
        self.lamb=0.75

        print('CONTEXTUAL + RNN NCL')

        return

    def _get_optimizer(self,lr=None,which_type=None):

        if which_type=='mcl':
            if lr is None: lr=self.lr
            return torch.optim.SGD(
                [p for p in self.model.mcl.parameters()]+[p for p in self.model.last.parameters()],lr=lr)
        elif which_type=='ac':
            if lr is None: lr=self.lr
            return torch.optim.SGD(
                [p for p in self.model.ac.parameters()]+[p for p in self.model.last.parameters()],lr=lr)

    def train(self,t,train,valid,args):
        # self.model=deepcopy(self.initial_model) # Restart model: isolate


        if t == 0: which_types = ['mcl']
        else: which_types = ['ac','mcl']

        for which_type in which_types:

            print('Training Type: ',which_type)

            best_loss=np.inf
            best_model=utils.get_model(self.model)
            lr=self.lr
            patience=self.lr_patience
            self.optimizer=self._get_optimizer(lr,which_type)

            # Loop epochs
            # self.nepochs=2
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                self.train_epoch(t,train,iter_bar,which_type)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,train,which_type)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/len(train),1000*self.sbatch*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,valid,which_type)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr,which_type)
                print()

            # Restore best
            utils.set_model_(self.model,best_model)

        return



    def train_epoch(self,t,data,iter_bar,which_type):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            outputs=self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s)
            output=outputs[t]
            loss=self.criterion(output,targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            if t>0 and which_type=='mcl':
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
                mask=self.model.ac.mask(task,s=self.smax)
                mask = torch.autograd.Variable(mask.data.clone(),requires_grad=False)
                # if step==0:
                    # print(mask.size)
                    # print(mask)
                # Commented out the masking operation - start (1 of 2)
                # for n,p in self.model.named_parameters():
                    # if n in rnn_weights:
                        # # print('n: ',n)
                        # # print('p: ',p.grad.size())
                        # p.grad.data*=self.model.get_view_for(n,mask)
                # Commented out the masking operation - end

            # Compensate embedding gradients
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den


            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

        return

    def eval(self,t,data,which_type,my_debug=0,input_tokens=None):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()


        for step, batch in enumerate(data):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch
            real_b=input_ids.size(0)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)
            outputs = self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s=self.smax)
            output=outputs[t]
            loss=self.criterion(output,targets)

            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy().item()*real_b
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=real_b
            
            if my_debug==1:
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
                    occ_output = self.model.forward(task,my_input_ids, my_segment_ids, my_input_mask, which_type, s=self.smax)[t]
                    occ_output = torch.nn.Softmax(dim=1)(occ_output)
                    actual_output = self.model.forward(task,input_ids[i:i+1,:], segment_ids[i:i+1,:], input_mask[i:i+1,:], which_type, s=self.smax)[t]
                    actual_output = torch.nn.Softmax(dim=1)(actual_output)
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
                        predictions = actual_pred
                        class_targets = targets[i:i+1]
                    else:
                        attributions_occ1 = torch.cat((attributions_occ1,attributions_occ1_b), axis=0)
                        predictions = torch.cat((predictions,actual_pred), axis=0)
                        class_targets = torch.cat((class_targets,targets[i:i+1]), axis=0)
            
            else:
                if step==0:
                    predictions = pred
                    class_targets = targets
                else:
                    predictions = torch.cat((predictions,pred), axis=0)
                    class_targets = torch.cat((class_targets,targets), axis=0)

            if my_debug==2:
                activations_b,mask = self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s=self.smax,my_debug=2)
                activations_b = activations_b.detach().cpu()
                if step==0:
                    activations = activations_b
                else:
                    activations = torch.cat((activations,activations_b), axis=0)
                

        # After looping through all batches
        if my_debug==1:    
            return class_targets, predictions, attributions_occ1
        if my_debug==2:    
            return class_targets, predictions, activations, mask

        return total_loss/total_num,total_acc/total_num
