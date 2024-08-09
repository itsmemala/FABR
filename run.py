import sys,os,argparse,time
import numpy as np
import pickle
import torch
from config import set_args
import utils
from utils import CPU_Unpickler
import attribution_utils
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import gc
from copy import deepcopy
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
tstart=time.time()

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"

# Arguments


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = set_args()

if args.output=='':
    args.output='FABR/res/'+args.experiment+'_'+args.approach+'_'+str(args.note)+'.txt'

performance_output=args.output+'_performance'
performance_output_forward=args.output+'_forward_performance'

# print('='*100)
# print('Arguments =')
# for arg in vars(args):
#     print('\t'+arg+':',getattr(args,arg))
# print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

########################################################################################################################

# Args -- EWC
# Source: https://github.com/ZixuanKe/PyContinual/blob/54dd15de566b110c9bc8d8316205de63a4805190/src/load_base_args.py
if 'bert_adapter' in args.backbone:
    args.apply_bert_output = True
    args.apply_bert_attention_output = True
########################################################################################################################

# Args -- Experiment
if args.experiment=='bert_dis':
    args.ntasks=6
    from dataloaders import bert_dis as dataloader
elif args.experiment=='bert_news':
    args.ntasks=6
    from dataloaders import bert_news as dataloader
elif args.experiment=='annomi':
    args.ntasks=6
    from dataloaders import bert_annomi as dataloader
elif args.experiment=='hwu64':
    args.ntasks=6
    from dataloaders import bert_hwu64 as dataloader


# Args -- Approach
if args.approach=='ctr':
    from approaches import bert_adapter_capsule_mask as approach
elif args.approach=='taskdrop':
    from approaches import taskdrop as approach
if args.backbone == 'bert_adapter':
    if args.baseline == 'ewc':
        from approaches import bert_adapter_ewc as approach
        from networks import bert_adapter as network
    elif args.baseline == 'ewc_freeze':
        from approaches import bert_adapter_ewc_freeze as approach
        from networks import bert_adapter as network
    elif args.baseline == 'ewc_ancl':
        from approaches import bert_adapter_ewc_ancl as approach
        from networks import bert_adapter as network
    elif args.baseline == 'lwf':
        from approaches import bert_adapter_lwf as approach
        from networks import bert_adapter as network
    elif args.baseline == 'lwf_ancl':
        from approaches import bert_adapter_lwf_ancl as approach
        from networks import bert_adapter as network
    elif args.baseline == 'seq' or args.baseline == 'mtl':
        from approaches import bert_adapter_seq as approach
        from networks import bert_adapter as network

# # Args -- Network
if 'ctr' in args.approach:
    from networks import bert_adapter_capsule_mask as network
elif args.approach=='mtl_bert_fine_tune' or args.approach=='bert_fine_tune':
    from networks import bert as network
#
# else:
#     raise NotImplementedError
#

########################################################################################################################

# Load
print('Load data...')
data,taskcla=dataloader.get(logger=logger,args=args)

print('\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(taskcla,args=args).cuda()

if 'ctr' in args.approach or 'bert_fine_tune' in args.approach or 'bert_adapter_ewc' in args.approach or 'bert_adapter_lwf' in args.approach or 'bert_adapter_seq' in args.approach or 'bert_adapter_mtl' in args.approach:
    appr=approach.Appr(net,logger=logger,taskcla=taskcla,args=args)
else:
    appr=approach.Appr(net,logger=logger,args=args)

# print('#trainable params:',sum(p.numel() for p in appr.model.parameters() if p.requires_grad))
# sys.exit()

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
f1=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

my_save_path = args.my_save_path

global_attr = {}

for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.transfer_acc==True and t>0: break # Only train on first task
    
    if args.start_at_task is not None and t<args.start_at_task:
        path = args.start_model_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt'
        list_of_lists = []
        with open(path, 'r') as f:
            for line in f:
                inner_list = [float(elt.strip()) for elt in line.split('\t')]
                list_of_lists.append(inner_list)
        for copy_j in range(len(list_of_lists[t])): # Copy task row from logged results
            acc[t,copy_j]=list_of_lists[t][copy_j]
        path = args.start_model_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'_f1.txt'
        list_of_lists = []
        with open(path, 'r') as f:
            for line in f:
                inner_list = [float(elt.strip()) for elt in line.split('\t')]
                list_of_lists.append(inner_list)
        for copy_j in range(len(list_of_lists[t])): # Copy task row from logged results
            f1[t,copy_j]=list_of_lists[t][copy_j]
        continue
    if t==args.start_at_task:
        # Restore checkpoints
        appr.model.load_state_dict(torch.load(args.start_model_path+'model'))
        appr.model_old = deepcopy(appr.model)
        appr.model_old.eval()
        utils.freeze_model(appr.model_old) # Freeze the weights
        with open(args.start_model_path+'fisher.pkl', 'rb') as handle:
            checkpoint_fisher = CPU_Unpickler(handle).load()
        with open(args.start_model_path+'fisher_old.pkl', 'rb') as handle:
            checkpoint_fisher_old = CPU_Unpickler(handle).load()
        with open(args.start_model_path+'fisher_for_loss.pkl', 'rb') as handle:
            checkpoint_fisher_for_loss = CPU_Unpickler(handle).load()
        appr.fisher, appr.fisher_old, appr.fisher_for_loss = {}, {}, {}
        for n,_ in appr.model.named_parameters(): # these will be None in case of non-fisher based approach
            if checkpoint_fisher is not None: appr.fisher[n] = checkpoint_fisher[n].cuda()
            if checkpoint_fisher_old is not None: appr.fisher_old[n] = checkpoint_fisher_old[n] #Note: This remains on cpu
            if checkpoint_fisher_for_loss is not None and len(checkpoint_fisher_for_loss.keys())>0: appr.fisher_for_loss[n] = checkpoint_fisher_for_loss[n].cuda() # Note this will be empty dict for ancl methods  
            

    if 'mtl' in args.approach:
        # Get data. We do not put it to GPU
        if t==0:
            train=data[t]['train']
            valid=data[t]['valid']
            num_train_steps=data[t]['num_train_steps']

        else:
            train = ConcatDataset([train,data[t]['train']])
            valid = ConcatDataset([valid,data[t]['valid']])
            num_train_steps+=data[t]['num_train_steps']
        task=t

        if t < len(taskcla)-1: continue #only want the last one

    else:
        # Get data
        train=data[t]['train']
        valid=data[t]['valid']
        num_train_steps=data[t]['num_train_steps']
        task=t

    train_sampler = RandomSampler(train)
    train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True)

    valid_sampler = SequentialSampler(valid)
    valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size, pin_memory=True)

    # Set task specific lr, if provided
    if args.custom_lr is not None:
        args.learning_rate = args.custom_lr[t]

    # Train
    if args.multi_plot_lail and t==args.break_after_task:
        # Checkpoint models and fisher
        # checkpoint_model = utils.get_model(appr.model)
        checkpoint_fisher, checkpoint_fisher_old, checkpoint_fisher_for_loss = {}, {}, {}
        for n,_ in appr.model.named_parameters():
            # checkpoint_fisher[n]=appr.fisher[n].clone().cpu() ## Changes to make space on GPU: #9
            # if appr.fisher_old is not None: checkpoint_fisher_old[n]=appr.fisher_old[n].clone().cpu() #Note: this will be none when only 1 task has been trained so far
            # checkpoint_fisher_for_loss[n]=appr.fisher_for_loss[n].clone().cpu()
            pass
        for lamb_i,plot_lamb in enumerate(args.plot_lambs):
            for thres_i,plot_thres in enumerate([0.5,0.6,0.7,0.8,0.9]):
                print('\nTraining for',lamb_i,thres_i,'\n')
                print('Researved:',torch.cuda.memory_reserved(0))
                print('Allocated:',torch.cuda.memory_allocated(0))
                print('Free:',torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0),'\n')
                appr.lamb = plot_lamb            
                appr.args.frel_cut = plot_thres
                # Train variant
                # appr.train(task,train_dataloader,valid_dataloader,args,num_train_steps,my_save_path,train,valid)
                # Save varients for plotting later
                if thres_i==0:
                    temp_model_path = my_save_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'_task'+str(t)+'lamsd_'+str(lamb_i)
                    # torch.save(appr.la_model, temp_model_path)
                    appr.plot_la_models[plot_lamb] = temp_model_path
                temp_model_path = my_save_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'_task'+str(t)+'mclmsd_'+str(lamb_i)+'_'+str(thres_i)
                # torch.save(utils.get_model(appr.model), temp_model_path)
                appr.plot_mcl_models[str(plot_lamb)+'_'+str(plot_thres)] = temp_model_path
                # Restore checkpoints
                # utils.set_model_(appr.model,checkpoint_model)
                for n,_ in appr.model.named_parameters():
                    # appr.fisher[n] = checkpoint_fisher[n].cuda() ## Changes to make space on GPU: #10
                    # if checkpoint_fisher_old!={}: appr.fisher_old[n] = checkpoint_fisher_old[n] #Note: This remains on cpu #Note: this will be none when only 1 task has been trained so far
                    # appr.fisher_for_loss[n] = checkpoint_fisher_for_loss[n].cuda()
                    pass
        # Multi-task model with same initialisation
        print('\nTraining Multi\n')
        appr.training_multi = True
        appr.lamb = 0
        appr.ce=torch.nn.CrossEntropyLoss()
        appr.args.use_rbs=False
        multi_train=data[0]['train']
        multi_valid=data[0]['valid']
        multi_num_train_steps=data[0]['num_train_steps']
        for temp_tid in range(1,t+1,1):
            multi_train = ConcatDataset([multi_train,data[temp_tid]['train']])
            multi_valid = ConcatDataset([multi_valid,data[temp_tid]['valid']])
            multi_num_train_steps+=data[temp_tid]['num_train_steps']
        multi_train_sampler = RandomSampler(multi_train)
        multi_train_dataloader = DataLoader(multi_train, sampler=multi_train_sampler, batch_size=args.train_batch_size, pin_memory=True)
        multi_valid_sampler = SequentialSampler(multi_valid)
        multi_valid_dataloader = DataLoader(multi_valid, sampler=multi_valid_sampler, batch_size=args.eval_batch_size, pin_memory=True)
        appr.train(task,multi_train_dataloader,multi_valid_dataloader,args,multi_num_train_steps,my_save_path,multi_train,multi_valid)
        appr.multi_model = appr.la_model
        appr.training_multi = False
    else:
        appr.train(task,train_dataloader,valid_dataloader,args,num_train_steps,my_save_path,train,valid)
    
    

    print('-'*100)

    # Plot loss along interpolation line
    if args.plot_lail and t==args.break_after_task:
        print('\nPlotting loss along interpolation line...\n')
        test=data[t]['test']
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size, pin_memory=True)
        past_test=data[0]['test']
        past_valid=data[0]['valid']
        for temp_tid in range(1,t,1):
            past_test = ConcatDataset([past_test,data[temp_tid]['test']])
            past_valid = ConcatDataset([past_valid,data[temp_tid]['valid']])
        past_test_sampler = RandomSampler(past_test)
        past_test_dataloader = DataLoader(past_test, sampler=past_test_sampler, batch_size=args.eval_batch_size, pin_memory=True)
        past_valid_sampler = SequentialSampler(past_valid)
        past_valid_dataloader = DataLoader(past_valid, sampler=past_valid_sampler, batch_size=args.eval_batch_size, pin_memory=True)
        fig_path = my_save_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'_task'+str(t)+'_interpolation_plot'
        appr.plot_loss_along_interpolation_line(network.Net(taskcla,args=args).cuda(),t,valid_dataloader,past_valid_dataloader,test_dataloader,past_test_dataloader,fig_path)
    
    # Test
    # for u in range(t+1):
    for u in range(len(taskcla)):
        
        if args.transfer_acc==False and u>t:
            continue
        if args.transfer_acc==True:
            eval_head=t # Eval using same head as the train data
        else:
            eval_head=u
        
        test=data[u]['test']
        test_sampler = SequentialSampler(test)
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size, pin_memory=True)

        test_loss,test_acc,test_f1=appr.eval(eval_head,test_dataloader)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss
        f1[t,u]=test_f1

    # Save
    print('Save at '+args.output)
    # np.savetxt(args.output,acc,'%.4f',delimiter='\t')
    np.savetxt(my_save_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',acc,'%.4f',delimiter='\t')
    np.savetxt(my_save_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'_f1.txt',f1,'%.4f',delimiter='\t')

    # appr.decode(train_dataloader)
    # break
    
    if t==args.break_after_task: # 1 implies only first 2 tasks
        torch.save(utils.get_model(appr.model), args.my_save_path+'model')
        with open(args.my_save_path+'fisher_old.pkl', 'wb') as fp:
            pickle.dump(appr.fisher_old, fp)
        with open(args.my_save_path+'fisher.pkl', 'wb') as fp:
            pickle.dump(appr.fisher, fp)
        with open(args.my_save_path+'fisher_for_loss.pkl', 'wb') as fp:
            pickle.dump(appr.fisher_for_loss, fp)
        break

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


# with open(performance_output,'w') as file:
    # if 'ncl' in args.approach  or 'mtl' in args.approach:
        # for j in range(acc.shape[1]):
            # file.writelines(str(acc[-1][j]) + '\n')

    # elif 'one' in args.approach:
        # for j in range(acc.shape[1]):
            # file.writelines(str(acc[j][j]) + '\n')


# with open(performance_output_forward,'w') as file:
    # if 'ncl' in args.approach  or 'mtl' in args.approach:
        # for j in range(acc.shape[1]):
            # file.writelines(str(acc[j][j]) + '\n')


########################################################################################################################
