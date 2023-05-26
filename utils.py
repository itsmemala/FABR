import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import pickle
import re

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=torch.autograd.Variable(x[b],volatile=False)
        target=torch.autograd.Variable(y[b],volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs=model.forward(images)
        loss=criterion(t,outputs[t],target)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/x.size(0)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################

def fisher_matrix_diag_bert(t,train,device,model,criterion,sbatch=20,scenario='til',imp='loss'):
    # Init
    fisher={}
    for n,p in model.named_parameters():
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

        if imp=='loss':
            loss=criterion(t,output,targets)
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
        # if 'output.adapter' in n or 'output.LayerNorm' in n:
            # print(fisher[n])
    
    # # Normalize by layer
    # layer_range = {}
    # layer_min = {}
    # for i in range(12):
        # wgts = torch.cat([
            # fisher['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.weight'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.bias'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.output.LayerNorm.weight'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.output.LayerNorm.bias'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.adapter.fc1.weight'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.adapter.fc1.bias'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.adapter.fc2.weight'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.adapter.fc2.bias'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.output.adapter.fc1.weight'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.output.adapter.fc1.bias'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.output.adapter.fc2.weight'].flatten()
            # ,fisher['bert.encoder.layer.'+str(i)+'.output.adapter.fc2.bias'].flatten()
        # ])
        # # wgts=torch.hstack(wgts).flatten()
        # assert len(wgts.shape)==1 # check that it is flattened
        # layer_min[str(i)] = torch.min(wgts)
        # layer_range[str(i)] = torch.max(wgts)-torch.min(wgts)
    
    # for n,_ in model.named_parameters():
        # if 'output.adapter' in n or 'output.LayerNorm' in n:
            # i = re.findall("layer\.(\d+)\.",n)[0]
            # fisher[n]=(fisher[n]-layer_min[i])/layer_range[i]
    return fisher

########################################################################################################################
#v11
def modified_fisher(fisher,fisher_old
                    ,model,model_old
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    # Adapt elasticity_down
    param_delta = []
    param_delta_sig = []
    for (name,param),(_,param_old) in zip(model.named_parameters(),model_old.named_parameters()):
        if 'output.adapter' in name or 'output.LayerNorm' in name:
            param_delta.append(torch.mean(torch.abs(param-param_old)).item())
            param_delta_sig.append(torch.mean(torch.sigmoid(torch.abs(param-param_old))).item())
    param_delta = np.mean(param_delta)
    param_delta_sig = np.mean(param_delta_sig)
    print('Avg param delta:',param_delta)
    print('Avg param delta sig:',param_delta_sig)
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]) # Relative importance
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic (i.e. increase fisher scaling)
            instability_check = lr*lamb*elasticity_down*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel>0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_down*fisher_rel[(fisher_rel>0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            instability_check = lr*lamb*elasticity_up*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel<=0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_up*fisher_rel[(fisher_rel<=0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
            
            modified_paramcount = torch.sum((fisher_rel<=0.5) & (instability_check==False))
            check_counter[n]=modified_paramcount
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    
    return modified_fisher
########################################################################################################################################
def modified_fisher_t0(fisher
                    ,model
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]) # Relative importance
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic (i.e. increase fisher scaling)
            instability_check = lr*lamb*elasticity_down*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel>0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_down*fisher_rel[(fisher_rel>0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            instability_check = lr*lamb*elasticity_up*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel<=0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_up*fisher_rel[(fisher_rel<=0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
            
            modified_paramcount = torch.sum((fisher_rel<=0.5) & (instability_check==False))
            check_counter[n]=modified_paramcount
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    
    return modified_fisher
    
