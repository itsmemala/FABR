# Version history for modified_fisher()

########################################################################################################################
def modified_fisher_t0(fisher
                    ,model
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,save_path):
    modified_fisher = {}
    
    check_counter = {}
    frozen_counter = {}
    rel_fisher_counter = {}
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]) # Relative importance
            rel_fisher_counter[n] = fisher_rel
            
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic (i.e. increase fisher scaling)
            instability_check = lr*lamb*elasticity_down*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel>0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_down*fisher_rel[(fisher_rel>0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            frozen_counter[n] = [torch.sum((fisher_rel>0.5) & (instability_check==True))]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            instability_check = lr*lamb*elasticity_up*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel<=0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_up*fisher_rel[(fisher_rel<=0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
            frozen_counter[n].append(torch.sum((fisher_rel<=0.5) & (instability_check==True)))
            
            modified_paramcount = torch.sum((fisher_rel<=0.5) & (instability_check==False))
            check_counter[n] = modified_paramcount
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    print('Frozen paramcount:',np.sum([v[0].cpu().numpy() for k,v in frozen_counter.items()]),np.sum([v[1].cpu().numpy() for k,v in frozen_counter.items()]))
    with open(save_path+'_frozen_paramcount.pkl', 'wb') as fp:
        pickle.dump(frozen_counter, fp)
    with open(save_path+'_relative_fisher.pkl', 'wb') as fp:
        pickle.dump(rel_fisher_counter, fp)
    
    return modified_fisher

########################################################################################################################
# v1
def modified_fisher(fisher,fisher_old):
    modified_fisher = {}
    
    check_counter = []
    
    for n in fisher.keys():
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        if 'output.adapter' in n:
            if torch.sum(fisher_old[n]) > torch.sum(fisher[n]): # Important for previous task only -> make it less elastic 
                modified_fisher[n] = 2*fisher_old[n]
                check_counter.append(1)
            else: # Other situations: Important for both or only new task or neither -> make it more elastic
                modified_fisher[n] = 0.5*fisher_old[n]
                check_counter.append(0)
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified percentage:',np.mean(check_counter))
    
    return modified_fisher
#########################################################
# v2
def modified_fisher(fisher,fisher_old,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    for n in fisher.keys():
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_gt = torch.gt(fisher_old[n],fisher[n])
            check_counter[n]=(torch.sum(fisher_gt))
            modified_fisher[n] = fisher_old[n]
            
            # Important for previous task only -> make it less elastic 
            modified_fisher[n][fisher_gt==True] = 2*fisher_old[n][fisher_gt==True]
            # Other situations: Important for both or only new task or neither -> make it more elastic
            modified_fisher[n][fisher_gt==False] = 0.5*fisher_old[n][fisher_gt==False]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    
    return modified_fisher
#########################################################
# v2.1, v2.2
def modified_fisher(fisher,fisher_old,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    for n in fisher.keys():
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_gt = torch.gt(fisher_old[n],fisher[n])
            check_counter[n]=(torch.sum(fisher_gt))
            modified_fisher[n] = fisher_old[n]
            
            # Important for previous task only -> make it less elastic 
            modified_fisher[n][fisher_gt==True] = 10*fisher_old[n][fisher_gt==True]
            # Other situations: Important for both or only new task or neither -> make it more elastic
            modified_fisher[n][fisher_gt==False] = 0.1*fisher_old[n][fisher_gt==False]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    
    return modified_fisher
    return modified_fisher
#########################################################
# v2.3, v2.4
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    for n in fisher.keys():
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_gt = torch.gt(fisher_old[n],fisher[n])
            check_counter[n]=(torch.sum(fisher_gt))
            modified_fisher[n] = fisher_old[n]
            
            # Important for previous tasks only -> make it less elastic 
            modified_fisher[n][fisher_gt==True] = elasticity_down*fisher_old[n][fisher_gt==True]
            # Other situations: Important for both or only new task or neither -> make it more elastic
            modified_fisher[n][fisher_gt==False] = elasticity_up*fisher_old[n][fisher_gt==False]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    
    return modified_fisher
#########################################################
# v3
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    for n in fisher.keys():
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_gt = torch.gt(fisher_old[n],fisher[n])
            fisher_lt = torch.lt(fisher_old[n],fisher[n])
            check_counter[n]=(torch.sum(fisher_gt))
            modified_fisher[n] = fisher_old[n]
            
            # Important for previous tasks only -> make it less elastic (i.e. increase fisher scaling)
            modified_fisher[n][fisher_gt==True] = elasticity_down*fisher_old[n][fisher_gt==True]
            # Important for both -> keep fisher 
            # Other situations: Important for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            modified_fisher[n][fisher_lt==True] = elasticity_up*fisher_old[n][fisher_lt==True]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    
    return modified_fisher
#########################################################
#v2.1-IC
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,lr,lamb,save_path):
    modified_fisher = {}
    
    check_counter = {}
    instability_counter = {}
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_gt = torch.gt(fisher_old[n],fisher[n])
            check_counter[n]=(torch.sum(fisher_gt))
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic
            instability_check = lr*lamb*elasticity_down*fisher_old[n][fisher_gt==True]
            instability_check = instability_check>1
            sum_instability_check = torch.sum(instability_check)
            if sum_instability_check>0:
                print('Unstable training!!')
            instability_counter[n]=[sum_instability_check]
            modified_fisher[n][fisher_gt==True] = elasticity_down*fisher_old[n][fisher_gt==True]
            
            # [2] Other situations: Important for both or only new task or neither -> make it more elastic
            instability_check = lr*lamb*elasticity_up*fisher_old[n][fisher_gt==False]
            instability_check = instability_check>1
            sum_instability_check = torch.sum(instability_check)
            if sum_instability_check>0:
                print('Unstable training!!')
            instability_counter[n].append(sum_instability_check)
            
            modified_fisher[n][fisher_gt==False] = elasticity_up*fisher_old[n][fisher_gt==False]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    with open(save_path+'_instability_paramcount.pkl', 'wb') as fp:
        pickle.dump(instability_counter, fp)
    
    return modified_fisher
#########################################################
# v4
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,lr,lamb,save_path):
    modified_fisher = {}
    
    check_counter = {}
    instability_counter = {}
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_gt = torch.gt(fisher_old[n],fisher[n])
            check_counter[n]=(torch.sum(fisher_gt))
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic
            instability_check = lr*lamb*elasticity_down*fisher_old[n]
            instability_check = instability_check>1
            
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_gt==True) & (instability_check==True)] = 1/(lr*lamb*elasticity_down)
            
            instability_check = lr*lamb*elasticity_down*fisher_old[n][fisher_gt==True]
            instability_check = instability_check>1
            sum_instability_check = torch.sum(instability_check)
            if sum_instability_check>0:
                print('Unstable training!!')
            instability_counter[n]=[sum_instability_check]
            modified_fisher[n][fisher_gt==True] = elasticity_down*fisher_old[n][fisher_gt==True]
            
            # [2] Other situations: Important for both or only new task or neither -> make it more elastic
            modified_fisher[n][fisher_gt==False] = elasticity_up*fisher_old[n][fisher_gt==False]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    with open(save_path+'_instability_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    
    return modified_fisher
#########################################################
#v5
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,lr,lamb,save_path):
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
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    # print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    # with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        # pickle.dump(check_counter, fp)
    
    return modified_fisher
#########################################################
#v6
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,lr,lamb,save_path):
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
            # [1a] If very high rel importance -> Reassign importance score; This essentially freezes the param
            fisher_old[n][fisher_rel>=0.9] = 1/(lr*lamb*elasticity_down*fisher_rel[fisher_rel>=0.9])
            # [1b] If medium rel importance
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    # print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    # with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        # pickle.dump(check_counter, fp)
    
    return modified_fisher
#########################################################
#v6.1
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,freeze_cutoff,lr,lamb,save_path):
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
            # [1a] If very high rel importance -> Reassign importance score; This essentially freezes the param
            fisher_old[n][fisher_rel>=freeze_cutoff] = 1/(lr*lamb*elasticity_down*fisher_rel[fisher_rel>=freeze_cutoff])
            # [1b] If medium rel importance
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    # print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    # with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        # pickle.dump(check_counter, fp)
    
    return modified_fisher
#########################################################
#v6.2
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,freeze_cutoff,lr,lamb,save_path):
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
            # Adjustment if very high rel importance in the top layer -> Reassign importance score; This essentially freezes the param
            if 'layer.11' in n:
                fisher_old[n][fisher_rel>=freeze_cutoff] = 1/(lr*lamb*elasticity_down*fisher_rel[fisher_rel>=freeze_cutoff])
            # Rest
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    # print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    # with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        # pickle.dump(check_counter, fp)
    
    return modified_fisher
########################################################################################################################
#v7, v9
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,freeze_cutoff,lr,lamb,save_path):
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
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    # print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    # with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        # pickle.dump(check_counter, fp)
    
    return modified_fisher
########################################################################################################################
#v8
def modified_fisher(fisher,fisher_old,train_loss_diff,elasticity_down,elasticity_up,freeze_cutoff,lr,lamb,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    elasticity_down*=train_loss_diff*10
    print('Loss based elasticity adaptation:',train_loss_diff,elasticity_down)
    
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
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    # print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    # with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        # pickle.dump(check_counter, fp)
    
    return modified_fisher
########################################################################################################################
#v8.1
def modified_fisher(fisher,fisher_old,train_loss_diff,elasticity_down,elasticity_up,freeze_cutoff,lr,lamb,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    if train_loss_diff<0:
        train_loss_diff=1    
    elasticity_down*=train_loss_diff
    print('Loss based elasticity adaptation:',train_loss_diff,elasticity_down)
    
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
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    # print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    # with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        # pickle.dump(check_counter, fp)
    
    return modified_fisher
########################################################################################################################
#v10
def modified_fisher(fisher,fisher_old,elasticity_down,elasticity_up,freeze_cutoff,lr,lamb,save_path):
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
########################################################################################################################
#v10 + avg param delta check
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

########################################################################################################################
#v13
def modified_fisher(fisher,fisher_old
                    ,train_f1,patience
                    ,model,model_old
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,save_path):
    modified_fisher = {}
    
    check_counter = {}
    
    # Adapt elasticity_down
    if len(train_f1)==50:
        train_f1_diff = train_f1[-1]-train_f1[0]
    else:
        train_f1_diff = train_f1[-patience]-train_f1[0]
    if train_f1_diff<0:
        train_f1_diff=1   
    elasticity_down*=train_f1_diff
    print('Elasticity adaptation:',train_f1_diff,elasticity_down)
    
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
########################################################################################################################
#v14
def modified_fisher(fisher,fisher_old
                    ,train_f1,patience
                    ,model,model_old
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,save_path):
    modified_fisher = {}
    
    check_counter = {}
    frozen_counter = {}
    rel_fisher_counter = {}
    
    # Adapt elasticity
    if len(train_f1)==50:
        train_f1_diff = (train_f1[-1]-train_f1[0])*100
    else:
        train_f1_diff = (train_f1[-patience]-train_f1[0])*100
        # train_f1_diff = (train_f1[-1]-train_f1[0])*100
    if train_f1_diff<1:
        train_f1_diff=1   
    elasticity_down=train_f1_diff
    elasticity_up=1/(train_f1_diff)
    print('Elasticity adaptation:',train_f1_diff,elasticity_down,elasticity_up)
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]) # Relative importance
            rel_fisher_counter[n] = fisher_rel
            
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic (i.e. increase fisher scaling)
            instability_check = lr*lamb*elasticity_down*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel>0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_down*fisher_rel[(fisher_rel>0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            frozen_counter[n] = [torch.sum((fisher_rel>0.5) & (instability_check==True))]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            instability_check = lr*lamb*elasticity_up*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel<=0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_up*fisher_rel[(fisher_rel<=0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
            frozen_counter[n].append(torch.sum((fisher_rel<=0.5) & (instability_check==True)))
            
            modified_paramcount = torch.sum((fisher_rel<=0.5) & (instability_check==False))
            check_counter[n]=modified_paramcount
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    print('Frozen paramcount:',np.sum([v[0].cpu().numpy() for k,v in frozen_counter.items()]),np.sum([v[1].cpu().numpy() for k,v in frozen_counter.items()]))
    with open(save_path+'_frozen_paramcount.pkl', 'wb') as fp:
        pickle.dump(frozen_counter, fp)
    with open(save_path+'_relative_fisher.pkl', 'wb') as fp:
        pickle.dump(rel_fisher_counter, fp)
    
    return modified_fisher
########################################################################################################################
#v15
def modified_fisher(fisher,fisher_old
                    ,train_f1,patience
                    ,model,model_old
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,save_path):
    modified_fisher = {}
    
    check_counter = {}
    frozen_counter = {}
    rel_fisher_counter = {}
    
    # Adapt elasticity
    if len(train_f1)==50:
        train_f1_diff = (train_f1[-1]-train_f1[0])*100
    else:
        train_f1_diff = (train_f1[-(patience+1)]-train_f1[0])*100
        # train_f1_diff = (train_f1[-1]-train_f1[0])*100
    if train_f1_diff<2:
        train_f1_diff=2
    elasticity_down=train_f1_diff
    elasticity_up=1/(train_f1_diff)
    print('Elasticity adaptation:',train_f1_diff,elasticity_down,elasticity_up)
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]) # Relative importance
            rel_fisher_counter[n] = fisher_rel
            
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic (i.e. increase fisher scaling)
            instability_check = lr*lamb*elasticity_down*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel>0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_down*fisher_rel[(fisher_rel>0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            frozen_counter[n] = [torch.sum((fisher_rel>0.5) & (instability_check==True))]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            instability_check = lr*lamb*elasticity_up*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel<=0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_up*fisher_rel[(fisher_rel<=0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
            frozen_counter[n].append(torch.sum((fisher_rel<=0.5) & (instability_check==True)))
            
            modified_paramcount = torch.sum((fisher_rel<=0.5) & (instability_check==False))
            check_counter[n]=modified_paramcount
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    print('Frozen paramcount:',np.sum([v[0].cpu().numpy() for k,v in frozen_counter.items()]),np.sum([v[1].cpu().numpy() for k,v in frozen_counter.items()]))
    with open(save_path+'_frozen_paramcount.pkl', 'wb') as fp:
        pickle.dump(frozen_counter, fp)
    with open(save_path+'_relative_fisher.pkl', 'wb') as fp:
        pickle.dump(rel_fisher_counter, fp)
    
    return modified_fisher
########################################################################################################################
#v16 (No instability adjustment)
def modified_fisher(fisher,fisher_old
                    ,train_f1,patience
                    ,model,model_old
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,save_path):
    modified_fisher = {}
    
    check_counter = {}
    frozen_counter = {}
    rel_fisher_counter = {}
    
    # Adapt elasticity
    if len(train_f1)==50:
        train_f1_diff = (train_f1[-1]-train_f1[0])*100
    else:
        train_f1_diff = (train_f1[-(patience+1)]-train_f1[0])*100
        # train_f1_diff = (train_f1[-1]-train_f1[0])*100
    if train_f1_diff<2:
        train_f1_diff=2
    elasticity_down=train_f1_diff
    elasticity_up=1/(train_f1_diff)
    print('Elasticity adaptation:',train_f1_diff,elasticity_down,elasticity_up)
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]) # Relative importance
            rel_fisher_counter[n] = fisher_rel
            
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic (i.e. increase fisher scaling)
            instability_check = lr*lamb*elasticity_down*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            # fisher_old[n][(fisher_rel>0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_down*fisher_rel[(fisher_rel>0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            frozen_counter[n] = [torch.sum((fisher_rel>0.5) & (instability_check==True))]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            instability_check = lr*lamb*elasticity_up*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            # fisher_old[n][(fisher_rel<=0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_up*fisher_rel[(fisher_rel<=0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
            frozen_counter[n].append(torch.sum((fisher_rel<=0.5) & (instability_check==True)))
            
            modified_paramcount = torch.sum((fisher_rel<=0.5) & (instability_check==False))
            check_counter[n]=modified_paramcount
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    print('Frozen paramcount:',np.sum([v[0].cpu().numpy() for k,v in frozen_counter.items()]),np.sum([v[1].cpu().numpy() for k,v in frozen_counter.items()]))
    with open(save_path+'_frozen_paramcount.pkl', 'wb') as fp:
        pickle.dump(frozen_counter, fp)
    with open(save_path+'_relative_fisher.pkl', 'wb') as fp:
        pickle.dump(rel_fisher_counter, fp)
    
    return modified_fisher
########################################################################################################################
# v17,v18,v19
def modified_fisher(fisher,fisher_old
                    ,train_f1,best_index
                    ,model,model_old
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,save_path):
    modified_fisher = {}
    
    check_counter = {}
    frozen_counter = {}
    rel_fisher_counter = {}
    
    # Adapt elasticity
    train_f1_diff = (train_f1[best_index]-train_f1[0])*100
    if train_f1_diff<2:
        train_f1_diff=2
    elasticity_down=train_f1_diff
    elasticity_up=1/(train_f1_diff)
    print('Elasticity adaptation:',train_f1_diff,elasticity_down,elasticity_up)
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n:
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]) # Relative importance
            rel_fisher_counter[n] = fisher_rel
            
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only -> make it less elastic (i.e. increase fisher scaling)
            instability_check = lr*lamb*elasticity_down*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel>0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_down*fisher_rel[(fisher_rel>0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel>0.5] = elasticity_down*fisher_rel[fisher_rel>0.5]*fisher_old[n][fisher_rel>0.5]
            frozen_counter[n] = [torch.sum((fisher_rel>0.5) & (instability_check==True))]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            instability_check = lr*lamb*elasticity_up*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            fisher_old[n][(fisher_rel<=0.5) & (instability_check==True)] = 1/(lr*lamb*elasticity_up*fisher_rel[(fisher_rel<=0.5) & (instability_check==True)])
            modified_fisher[n][fisher_rel<=0.5] = elasticity_up*fisher_rel[fisher_rel<=0.5]*fisher_old[n][fisher_rel<=0.5]
            frozen_counter[n].append(torch.sum((fisher_rel<=0.5) & (instability_check==True)))
            
            modified_paramcount = torch.sum((fisher_rel<=0.5) & (instability_check==False))
            check_counter[n]=modified_paramcount
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    print('Frozen paramcount:',np.sum([v[0].cpu().numpy() for k,v in frozen_counter.items()]),np.sum([v[1].cpu().numpy() for k,v in frozen_counter.items()]))
    with open(save_path+'_frozen_paramcount.pkl', 'wb') as fp:
        pickle.dump(frozen_counter, fp)
    with open(save_path+'_relative_fisher.pkl', 'wb') as fp:
        pickle.dump(rel_fisher_counter, fp)
    
    return modified_fisher

########################################################################################################################
# v17.2 (AAAI expts), v20
def modified_fisher(fisher,fisher_old
                    ,train_f1,best_index
                    ,model,model_old
                    ,elasticity_down,elasticity_up
                    ,freeze_cutoff
                    ,lr,lamb
                    ,grad_dir_lastart=None,grad_dir_laend=None,lastart_fisher=None
                    ,save_path=''):
    frel_cut = 0.5
    modified_fisher = {}
    
    check_counter = {}
    frozen_counter = {}
    rel_fisher_counter = {}
    check_graddir_counter = {}
    
    # Adapt elasticity
    if best_index>=0:
        train_f1_diff = (train_f1[best_index]-train_f1[0])*100
        if train_f1_diff<2:
            train_f1_diff=2
    else:
        train_f1_diff=1
    elasticity_down=train_f1_diff
    elasticity_up=1/(train_f1_diff)
    print('Elasticity adaptation:',train_f1_diff,elasticity_down,elasticity_up)
    
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        
        if 'output.adapter' in n or 'output.LayerNorm' in n: #or 'last' in n:
            # if 'last' in n:
                # print('calculating for last layer...\n\n')
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]) # Relative importance
            rel_fisher_counter[n] = fisher_rel
            
            if grad_dir_lastart is not None:
                alpha_delta = (fisher[n] - lastart_fisher[n])
                graddir = torch.abs(grad_dir_laend[n] - grad_dir_lastart[n])
                check_graddir = torch.nan_to_num((graddir-alpha_delta)/graddir,nan=0,posinf=0,neginf=0)
                check_graddir = check_graddir>1
                # print(torch.sum(check_graddir))
                # print(check_graddir.shape,graddir.shape,fisher_rel.shape)
            
            modified_fisher[n] = fisher_old[n]
            
            # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
            instability_check = lr*lamb*elasticity_down*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            # fisher_old[n][(fisher_rel>frel_cut) & (instability_check==True)] = 1/(lr*lamb*elasticity_down*fisher_rel[(fisher_rel>frel_cut) & (instability_check==True)])
            # frozen_counter[n] = [torch.sum((fisher_rel>frel_cut) & (instability_check==True))]
            if grad_dir_lastart is not None:
                modified_fisher[n][fisher_rel>frel_cut] = elasticity_down*fisher_rel[fisher_rel>frel_cut]*fisher_old[n][fisher_rel>frel_cut]
            else:
                modified_fisher[n][fisher_rel>frel_cut] = elasticity_down*fisher_rel[fisher_rel>frel_cut]*fisher_old[n][fisher_rel>frel_cut]
            
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            instability_check = lr*lamb*elasticity_up*fisher_rel*fisher_old[n]
            instability_check = instability_check>1
            # Adjustment if out of stability region -> Reassign importance score; This essentially freezes the param
            # fisher_old[n][(fisher_rel<=frel_cut) & (instability_check==True)] = 1/(lr*lamb*elasticity_up*fisher_rel[(fisher_rel<=frel_cut) & (instability_check==True)])
            # frozen_counter[n].append(torch.sum((fisher_rel<=frel_cut) & (instability_check==True)))
            if grad_dir_lastart is not None:
                modified_fisher[n][(fisher_rel<=frel_cut) & (check_graddir==False)] = elasticity_up*fisher_rel[(fisher_rel<=frel_cut) & (check_graddir==False)]*fisher_old[n][(fisher_rel<=frel_cut) & (check_graddir==False)]
            else:
                modified_fisher[n][fisher_rel<=frel_cut] = elasticity_up*fisher_rel[fisher_rel<=frel_cut]*fisher_old[n][fisher_rel<=frel_cut]
            
            # modified_paramcount = torch.sum((fisher_rel<=frel_cut) & (instability_check==False))
            modified_paramcount = torch.sum((fisher_rel<=frel_cut))
            check_counter[n]=modified_paramcount
            if grad_dir_lastart is not None:
                check_graddir_counter[n]=torch.sum((fisher_rel<=frel_cut) & (check_graddir==False))
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('All KT paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    print('Positive KT paramcount:',np.sum([v.cpu().numpy() for k,v in check_graddir_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    # print('Frozen paramcount:',np.sum([v[0].cpu().numpy() for k,v in frozen_counter.items()]),np.sum([v[1].cpu().numpy() for k,v in frozen_counter.items()]))
    # with open(save_path+'_frozen_paramcount.pkl', 'wb') as fp:
        # pickle.dump(frozen_counter, fp)
    with open(save_path+'_relative_fisher.pkl', 'wb') as fp:
        pickle.dump(rel_fisher_counter, fp)
    
    return modified_fisher