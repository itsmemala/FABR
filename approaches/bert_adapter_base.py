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
import torch.nn.functional as F
import nlp_data_utils as data_utils
from copy import deepcopy
sys.path.append("./approaches/")
from .contrastive_loss import SupConLoss, CRDLoss
from .buffer import Buffer as Buffer
from .buffer import Attr_Buffer as Attr_Buffer
from .buffer import RRR_Buffer as RRR_Buffer


def log_softmax(idrandom,t,x,class_counts=None):
    # print('This is my custom function.')
    # print(x[0,:])
    if class_counts is None:
        class_counts=1
    # TODO: Replace 5,30 with arg in case number of classes per task is a variable
    my_lambda = utils.get_my_lambda(idrandom,t,class_counts)
    # classes_seen = t*5
    # classes_cur = 5
    # classes_later = 30-(classes_seen+classes_cur)
    # my_lambda = torch.cat([torch.ones(classes_seen)*0,torch.ones(classes_cur)*torch.tensor(class_counts),torch.zeros(classes_later)], dim=0).cuda()
    assert len(my_lambda)==x.shape[1]
    softmax = my_lambda*torch.exp(x) / torch.sum(my_lambda*torch.exp(x), dim=1, keepdim=True)
    softmax_clamp = softmax.clamp(min=1e-16) # Clamp the zeros to avoid nan gradients
    return torch.log(softmax_clamp)

def MyBalancedCrossEntropyLoss(idrandom):
    def my_bal_ce(t, outputs, targets, class_counts=None):
        return torch.nn.functional.nll_loss(log_softmax(idrandom,t,outputs,class_counts), targets)
    return my_bal_ce

class Appr(object):

    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x


    def __init__(self,model,logger,taskcla, args=None):

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        logger.info("device: {} n_gpu: {}".format(
            self.device, self.n_gpu))
        self.sup_con = SupConLoss(temperature=args.temp,base_temperature=args.base_temp)

        # shared ==============
        self.model=model
        self.model_old=None
        self.model_aux=None
        self.la_model,self.mcl_model,self.plot_la_models,self.plot_mcl_models,self.multi_model,self.plot_model=None,None,{},{},None,None
        self.training_multi=None
        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.args=args
        if args.experiment=='annomi' and args.use_cls_wgts==True:
            print('Using cls wgts')
            class_weights = [0.41, 0.89, 0.16] #'change': 0, 'sustain': 1, 'neutral': 2
            class_weights = torch.FloatTensor(class_weights).cuda()
            self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif args.scenario=='cil' and args.use_rbs==False:
            self.ce=torch.nn.CrossEntropyLoss()
            # self.ce2 = MyBalancedCrossEntropyLoss()
        elif args.scenario=='cil' and args.use_rbs:
            self.ce = MyBalancedCrossEntropyLoss(self.args.idrandom)
        else:
            self.ce=torch.nn.CrossEntropyLoss()
        self.taskcla = taskcla
        self.logger = logger

        if args.baseline=='ewc' or args.baseline=='ewc_freeze' or args.baseline=='ewc_ancl':
            if self.args.use_ind_lamb_max==True:
                self.lamb={}
            else:
                self.lamb=args.lamb                  # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
            self.fisher=None
            self.fisher_old=None
            self.fisher_for_loss=None
            self.alpha_lamb=args.alpha_lamb
        
        if args.baseline=='lwf' or args.baseline=='lwf_ancl':
            self.lamb=args.lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
            self.alpha_lamb=args.alpha_lamb
            self.lwf_T=args.lwf_T
        
        if args.baseline=='ewc_fabr':
            self.lamb=args.lamb # Remove if not using ewc loss
            self.buffer = Attr_Buffer(self.args.buffer_size, 'cpu') # using cpu to avoid cuda memory err
            self.mse = torch.nn.MSELoss()

        #OWM ============
        if args.baseline=='owm':
            dtype = torch.cuda.FloatTensor  # run on GPU
            self.P1 = torch.autograd.Variable(torch.eye(self.args.bert_adapter_size).type(dtype), volatile=True) #inference only
            self.P2 = torch.autograd.Variable(torch.eye(self.args.bert_adapter_size).type(dtype), volatile=True)

        #UCL ======================
        if  args.baseline=='ucl':
            self.saved = 0
            self.beta = args.beta
            self.model=model
            self.model_old = deepcopy(self.model)

        if args.baseline=='one':
            self.model=model
            self.initial_model=deepcopy(model)

        if  args.baseline=='derpp':
            # self.buffer = Buffer(self.args.buffer_size, self.device)
            self.buffer = Buffer(self.args.buffer_size, 'cpu') # using cpu to avoid cuda memory err
            self.mse = torch.nn.MSELoss()

        if  args.baseline=='derpp_fabr':
            self.buffer = Attr_Buffer(self.args.buffer_size, 'cpu') # using cpu to avoid cuda memory err
            self.mse = torch.nn.MSELoss()
        
        if  args.baseline=='replay':
            self.buffer = Buffer(self.args.buffer_size, 'cpu') # using cpu to avoid cuda memory err
            self.mse = torch.nn.MSELoss()
        
        if  args.baseline=='rrr':
            self.buffer = RRR_Buffer(self.args.buffer_size, 'cpu') # using cpu to avoid cuda memory err
            self.mse = torch.nn.MSELoss()

        if  args.baseline=='gem':
            self.buffer = Buffer(self.args.buffer_size, self.device)
            # Allocate temporary synaptic memory
            self.grad_dims = []
            for pp in model.parameters():
                self.grad_dims.append(pp.data.numel())

            self.grads_cs = []
            self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)

        if  args.baseline=='a-gem':
            self.buffer = Buffer(self.args.buffer_size, self.device)
            self.grad_dims = []
            for param in self.model.parameters():
                self.grad_dims.append(param.data.numel())
            self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
            self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

        if  args.baseline=='l2':
            self.lamb=self.args.lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
            self.regularization_terms = {}
            self.task_count = 0
            self.online_reg = False  # True: There will be only one importance matrix and previous model parameters
                                    # False: Each task has its own importance matrix and model parameters
        print('BERT ADAPTER BASE')

        return

    def sup_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets,t):
        if self.args.sup_head:
            outputs = torch.cat([output.clone().unsqueeze(1), output.clone().unsqueeze(1)], dim=1)
        else:
            outputs = torch.cat([pooled_rep.clone().unsqueeze(1), pooled_rep.clone().unsqueeze(1)], dim=1)

        loss = self.sup_con(outputs, targets,args=self.args)
        return loss


    def order_generation(self,t):
        orders = []
        nsamples = t
        for n in range(self.args.naug):
            if n == 0: orders.append([pre_t for pre_t in range(t)])
            elif nsamples>=1:
                orders.append(random.Random(self.args.seed).sample([pre_t for pre_t in range(t)],nsamples))
                nsamples-=1
        return orders

    def idx_generator(self,bsz):
        #TODO: why don't we generate more?
        ls,idxs = [],[]
        for n in range(self.args.ntmix):
            if self.args.tmix:
                if self.args.co:
                    mix_ = np.random.choice([0, 1], 1)[0]
                else:
                    mix_ = 1

                if mix_ == 1:
                    l = np.random.beta(self.args.alpha, self.args.alpha)
                    if self.args.separate_mix:
                        l = l
                    else:
                        l = max(l, 1-l)
                else:
                    l = 1
                idx = torch.randperm(bsz) # Note I currently do not havce unsupervised data
            ls.append(l)
            idxs.append(idx)

        return idxs,ls


    def f1_compute_fn(self,y_true, y_pred,average):
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred, average=average, labels=np.unique(y_true))

    def criterion(self,t,output,targets,class_counts=None,phase=None, outputs_cur1=None, targets_old=None, outputs_cur2=None, targets_aux=None):
        # Regularization for all previous tasks
        loss_reg=0
        loss_ancl_reg=0
        if t>0:
            if self.args.ancl==False and self.args.lwf_ancl==False and self.args.lwf==False:
                if (phase=='fo' and self.args.no_reg_in_LA==True) or phase is None:
                    pass
                else:
                    fisher = self.fisher if self.fisher_for_loss is None else self.fisher_for_loss # baseline:self.fisher, LA:self.fisher_for_loss
                    if self.args.use_ind_lamb_max==True:
                        for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                            loss_reg+=torch.sum(self.lamb[name]*fisher[name]*(param_old-param).pow(2))/2
                    else:
                        for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                            loss_reg+=torch.sum(fisher[name]*(param_old-param).pow(2))/2
                        loss_reg = self.lamb*loss_reg
            elif self.args.ancl==True:
                if phase=='fo' or phase is None:
                    pass
                else:
                    for (name,param),(_,param_old), (_,param_aux) in zip(self.model.named_parameters(),self.model_old.named_parameters(),self.model_aux.named_parameters()):
                        loss_reg+=torch.sum(self.fisher_old[name]*(param_old-param).pow(2))/2
                        loss_ancl_reg+=torch.sum(self.fisher[name]*(param_aux-param).pow(2))/2
                    # print('loss_reg:',loss_reg,' loss_ancl_reg:',loss_ancl_reg)
                    loss_reg = self.lamb*loss_reg
                    loss_ancl_reg = self.alpha_lamb*loss_ancl_reg
            elif self.args.lwf_ancl==True:
                if phase=='fo' or phase is None:
                    pass
                else:
                    if 'til' in self.args.scenario:
                        loss_reg = self.lamb*self.lwf_cross_entropy(torch.cat(outputs_cur1, dim=1),
                                                            torch.cat(targets_old, dim=1), exp=1.0 / self.lwf_T)
                    else:
                        loss_reg = self.lamb*self.lwf_cross_entropy(outputs_cur1,
                                                        targets_old, exp=1.0 / self.lwf_T)
                    loss_ancl_reg = self.alpha_lamb*self.lwf_cross_entropy(outputs_cur2,
                                                   targets_aux, exp=1.0 / self.lwf_T)
            elif self.args.lwf==True:
                if 'til' in self.args.scenario:
                    loss_reg = self.lamb*self.lwf_cross_entropy(torch.cat(outputs_cur1, dim=1),
                                                        torch.cat(targets_old, dim=1), exp=1.0 / self.lwf_T)
                else:
                    loss_reg = self.lamb*self.lwf_cross_entropy(outputs_cur1,
                                                    targets_old, exp=1.0 / self.lwf_T)

        # assert self.ce(output,targets)==self.ce2(output,targets)

        if 'cil' in self.args.scenario and self.args.use_rbs:
            loss_ce = self.ce(t,output,targets,class_counts)
        else:
            loss_ce = self.ce(output,targets)

        if self.args.use_l1==True:
            # print('using L1')
            loss_l1=0
            if phase is None:
                pass
            else:
                for name,param in self.model.named_parameters():
                    loss_l1+=torch.sum(torch.abs(param))
            return loss_ce+loss_reg+self.args.l1_lamb*loss_l1
        
        elif self.args.use_l2==True:
            loss_l2=0
            if phase is None:
                pass
            else:
                for name,param in self.model.named_parameters():
                    loss_l2+=torch.sum(torch.square(param))
            return loss_ce+loss_reg+self.args.l2_lamb*loss_l2
        
        elif self.args.ancl==True:
            return loss_ce+loss_reg+loss_ancl_reg  
        
        elif self.args.lwf_ancl==True:
            return loss_ce+loss_reg+loss_ancl_reg
        elif self.args.lwf==True:
            return loss_ce+loss_reg  
        
        else:
            return loss_ce+loss_reg
    
    def lwf_cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce
    
    def criterion_fabr(self,t,output,targets,attributions,buffer_attributions):
        # Feature Attribution Based Regularization
        loss_fabr=0
        if t>0:
            pass

        return self.args.lamba*loss_fabr

