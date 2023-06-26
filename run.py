import sys,os,argparse,time
import numpy as np
import pickle
import torch
from config import set_args
import utils
import attribution_utils
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
tstart=time.time()

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

# Args -- DER++, EWC, REPLAY
# Source: https://github.com/ZixuanKe/PyContinual/blob/54dd15de566b110c9bc8d8316205de63a4805190/src/load_base_args.py
if 'bert_adapter' in args.backbone:
    args.apply_bert_output = True
    args.apply_bert_attention_output = True
if args.baseline == 'derpp' or args.baseline == 'derpp_fabr' or args.baseline == 'replay' or args.baseline == 'rrr':
    args.buffer_size = 28
    args.buffer_percent = 0.02
    args.alpha = 0.5
    args.beta = 0.5
if args.baseline=='derpp_fabr' or args.baseline == 'rrr':
    args.lfa_lambda = 0.00001
########################################################################################################################

########################################################################################################################

# Args -- CTR
# Source: https://github.com/ZixuanKe/PyContinual/blob/54dd15de566b110c9bc8d8316205de63a4805190/src/load_base_args.py
# if args.baseline == 'ctr':
if args.approach=='ctr':
    args.apply_bert_output = True
    args.apply_bert_attention_output = True
    args.build_adapter_capsule_mask = True
    args.apply_one_layer_shared = True
    args.use_imp = True
    args.transfer_route = True
    args.share_conv = True
    args.larger_as_share = True
    # args.adapter_size = True

########################################################################################################################

# Args -- Experiment
if args.experiment=='w2v':
    from dataloaders import w2v as dataloader
elif args.experiment=='bert':
    from dataloaders import bert as dataloader
elif  args.experiment=='bert_gen_hat':
    from dataloaders import bert_gen_hat as dataloader
elif  args.experiment=='bert_gen' or args.experiment=='bert_gen_single':
    from dataloaders import bert_gen as dataloader
elif args.experiment=='bert_sep':
    from dataloaders import bert_sep as dataloader
# Added this to mirror Taskdrop setup
elif args.experiment=='bert_dis':
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
if args.approach=='bert_lstm_ncl' or args.approach=='bert_gru_ncl' or args.approach=='mtl_gru':
    from approaches import bert_rnn_ncl as approach
elif args.approach=='bert_lstm_kan_ncl' or args.approach=='bert_gru_kan_ncl':
    from approaches import bert_rnn_kan_ncl as approach
elif args.approach=='bert_mlp_ncl':
    from approaches import bert_mlp_ncl as approach
elif args.approach=='ctr':
    from approaches import bert_adapter_capsule_mask as approach
elif args.approach=='taskdrop':
    from approaches import taskdrop as approach
elif args.approach=='mtl_bert_fine_tune' or args.approach=='bert_fine_tune':
    from approaches import bert_mtl as approach
if args.backbone == 'bert_adapter':
    if args.baseline == 'derpp':
        from approaches import bert_adapter_derpp as approach
        from networks import bert_adapter as network
    elif args.baseline == 'derpp_fabr':
        from approaches import bert_adapter_derpp_fabr as approach
        from networks import bert_adapter as network
    elif args.baseline == 'ewc':
        from approaches import bert_adapter_ewc as approach
        from networks import bert_adapter as network
    # elif args.baseline == 'ewc_fabr':
        # from approaches import bert_adapter_ewc_fabr as approach
        # from networks import bert_adapter as network
    elif args.baseline == 'ewc_freeze':
        from approaches import bert_adapter_ewc_freeze as approach
        from networks import bert_adapter as network
    elif args.baseline == 'seq' or args.baseline == 'mtl':
        from approaches import bert_adapter_seq as approach
        from networks import bert_adapter as network
    elif args.baseline == 'replay':
        from approaches import bert_adapter_replay as approach
        from networks import bert_adapter as network
    elif args.baseline == 'rrr':
        from approaches import bert_adapter_rrr as approach
        from networks import bert_adapter as network

# # Args -- Network
if 'bert_lstm_kan' in args.approach:
    from networks import bert_lstm_kan as network
elif 'bert_lstm' in args.approach:
    from networks import bert_lstm as network
if 'bert_gru_kan' in args.approach:
    from networks import bert_gru_kan as network
elif 'bert_gru' in args.approach or args.approach=='mtl_gru':
    from networks import bert_gru as network
elif 'bert_mlp' in args.approach:
    from networks import bert_mlp as network
elif 'ctr' in args.approach:
    from networks import bert_adapter_capsule_mask as network
elif 'taskdrop' in args.approach:
    from networks import taskdrop as network
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

if 'ctr' in args.approach or 'bert_fine_tune' in args.approach or 'bert_adapter_ewc' in args.approach or 'bert_adapter_seq' in args.approach or 'bert_adapter_mtl' in args.approach:
    appr=approach.Appr(net,logger=logger,taskcla=taskcla,args=args)
else:
    appr=approach.Appr(net,logger=logger,args=args)

# print('#trainable params:',sum(p.numel() for p in appr.model.parameters() if p.requires_grad))
# sys.exit()

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
f1=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)

# my_save_path = '/content/gdrive/MyDrive/s200_kan_myocc_attributions_bymask/' #NoMask
# my_save_path = '/content/gdrive/MyDrive/s200_kan_myocc_attributions_lfa/s200r - EWC Adapter BERT (train_attributions)/'
# my_save_path = '/content/gdrive/MyDrive/s200_kan_myocc_attributions_lfa/'
my_save_path = args.my_save_path

global_attr = {}

for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.transfer_acc==True and t>0: break # Only train on first task
    

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
    train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_sampler = SequentialSampler(valid)
    valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size)

    with open(my_save_path+str(args.note)+'_seed'+str(args.seed)+"_inputtokens_task"+str(t)+".txt", "wb") as internal_filename:
        pickle.dump(data[t]['train_tokens'], internal_filename)
    with open(my_save_path+str(args.note)+'_seed'+str(args.seed)+"_inputtokens_task"+str(t)+"_test.txt", "wb") as internal_filename:
        pickle.dump(data[t]['test_tokens'], internal_filename)

    # Train
    if args.lfa is None: # No attribution calculation at train time
        if 'ctr' in args.approach or 'bert_fine_tune' in args.approach:
            appr.train(task,train_dataloader,valid_dataloader,args,num_train_steps,my_save_path)
        elif 'bert_adapter_derpp' in args.approach or 'bert_adapter_ewc' in args.approach or 'bert_adapter_replay' in args.approach or 'bert_adapter_rrr' in args.approach or 'bert_adapter_seq' in args.approach or 'bert_adapter_mtl' in args.approach:
            appr.train(task,train_dataloader,valid_dataloader,args,num_train_steps,my_save_path,train,valid)
        else:
            appr.train(task,train_dataloader,valid_dataloader,args,my_save_path)
    elif args.lfa=='test0':
        appr.train(task,train_dataloader,valid_dataloader,args,my_save_path,data[t]['train_tokens'],global_attr=None) # Only local attributions
    else:
        if t==0:
            appr.train(task,train_dataloader,valid_dataloader,args,my_save_path) # No attribution calculation at train time
        else:
            appr.train(task,train_dataloader,valid_dataloader,args,my_save_path,data[t]['train_tokens'],global_attr) # Use global attributions from memory
    print('-'*100)

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
        test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

        if 'kan' in args.approach:
            test_loss,test_acc,test_f1=appr.eval(eval_head,test_dataloader,'mcl')
        else:
            test_loss,test_acc,test_f1=appr.eval(eval_head,test_dataloader)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss
        f1[t,u]=test_f1
        
        # # Load saved model and check that test acc and loss are the same
        # if 'ewc' in args.approach:
            # pass # Can't do this for ewc since old_model causes err #TODO: Fix this
        # else:
            # if 'ctr' in args.approach or 'bert_fine_tune' in args.approach or 'bert_adapter_ewc' in args.approach:
                # check_appr=approach.Appr(net,logger=logger,taskcla=taskcla,args=args)
            # else:
                # check_appr=approach.Appr(net,logger=logger,args=args)
            # check_appr.model.load_state_dict(torch.load(my_save_path+str(args.note)+'_seed'+str(args.seed)+'_model'+str(t)))
            # check_loss,check_acc,check_f1=check_appr.eval(eval_head,test_dataloader,'mcl')
            # if args.approach=='ctr':
                # print(check_loss,test_loss)
                # print(check_acc,test_acc)
                # print(check_f1,test_f1)
            # else:
                # #TODO: Check why check_loss==test_loss fails for ctr
                # assert check_loss==test_loss and check_acc==test_acc and check_f1==test_f1
                   
        if args.save_metadata=='all' or args.save_metadata=='train_attributions':
            # Train data attributions
            # Calculate attributions on all previous tasks and current task after training
            train = data[u]['train']
            train_sampler = SequentialSampler(train) # Retain the order of the dataset, i.e. no shuffling
            train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)
            if args.approach=='bert_adapter_ewc_fabr':
                targets, predictions, attributions = appr.get_attributions(eval_head,train_dataloader)
            elif args.approach=='bert_fine_tune' or args.approach=='bert_adapter_rrr' or args.approach=='ctr' or args.approach=='bert_adapter_seq':
                targets, predictions, attributions = appr.get_attributions(eval_head,train_dataloader,input_tokens=data[u]['train_tokens'])
            elif 'kan' in args.approach:
                targets, predictions, attributions = appr.eval(eval_head,train_dataloader,'mcl',my_debug=1,input_tokens=data[u]['train_tokens'])
            np.savez_compressed(my_save_path+str(args.note)+'_seed'+str(args.seed)+'_attributions_model'+str(t)+'task'+str(u)
                                ,targets=targets.cpu()
                                ,predictions=predictions.cpu()
                                ,attributions=attributions.cpu()
                                )
        
        if args.save_metadata=='all' or args.save_metadata=='test_attributions':        
            # Test data attributions
            # Calculate attributions on current task after training
            if args.approach=='bert_fine_tune' or args.approach=='bert_adapter_rrr' or args.approach=='ctr' or args.approach=='bert_adapter_seq':
                targets, predictions, attributions = appr.get_attributions(eval_head,test_dataloader,input_tokens=data[u]['test_tokens'])
            elif 'kan' in args.approach:
                targets, predictions, attributions = appr.eval(eval_head,test_dataloader,'mcl',my_debug=1,input_tokens=data[u]['test_tokens'])
            np.savez_compressed(my_save_path+str(args.note)+'_seed'+str(args.seed)+'_testattributions_model'+str(t)+'task'+str(u)
                                ,targets=targets.cpu()
                                ,predictions=predictions.cpu()
                                ,attributions=attributions
                                )
        
        if args.save_metadata=='all' and 'kan' in args.approach:
            # Train data activations # Only for KAN
            targets, predictions, activations, mask = appr.eval(eval_head,train_dataloader,'mcl',my_debug=2,input_tokens=data[u]['train_tokens'])
            np.savez_compressed(my_save_path+str(args.note)+'_seed'+str(args.seed)+'_activations_model'+str(t)+'task'+str(u)
                                ,activations=activations
                                ,mask=mask.detach().cpu()
                                )
            
            # Test data activations # Only for KAN
            targets, predictions, activations, mask = appr.eval(eval_head,test_dataloader,'mcl',my_debug=2,input_tokens=data[u]['test_tokens'])
            np.savez_compressed(my_save_path+str(args.note)+'_seed'+str(args.seed)+'_testactivations_model'+str(t)+'task'+str(u)
                                ,activations=activations
                                ,mask=mask.detach().cpu()
                                )

        if (args.lfa is not None) and ('test' not in args.lfa):
            # Save global attributions for using when training the next task
            if u==t: # Only for the current task
                # global_attr = None
                global_attr[t] = attribution_utils.aggregate_local_to_global(attributions_occ1,predictions.cpu(),targets.cpu(),data[t]['train_tokens'])

    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f',delimiter='\t')
    np.savetxt(my_save_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'.txt',acc,'%.4f',delimiter='\t')
    np.savetxt(my_save_path+args.experiment+'_'+args.approach+'_'+str(args.note)+'_seed'+str(args.seed)+'_f1.txt',f1,'%.4f',delimiter='\t')

    # appr.decode(train_dataloader)
    # break
    
    # if t==1: # Only first 2 tasks
        # break

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


with open(performance_output,'w') as file:
    if 'ncl' in args.approach  or 'mtl' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[-1][j]) + '\n')

    elif 'one' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[j][j]) + '\n')


with open(performance_output_forward,'w') as file:
    if 'ncl' in args.approach  or 'mtl' in args.approach:
        for j in range(acc.shape[1]):
            file.writelines(str(acc[j][j]) + '\n')


########################################################################################################################
