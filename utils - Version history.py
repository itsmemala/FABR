# Version history for modified_fisher()

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
# v2.4
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
            # instability_check = lr*lamb*elasticity_up*fisher_old[n][fisher_gt==False]
            # instability_check = instability_check>1
            # sum_instability_check = torch.sum(instability_check)
            # if sum_instability_check>0:
                # print('Unstable training!!')
            # instability_counter[n].append(sum_instability_check)
            
            modified_fisher[n][fisher_gt==False] = elasticity_up*fisher_old[n][fisher_gt==False]
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    print('Modified paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    with open(save_path+'_instability_paramcount.pkl', 'wb') as fp:
        pickle.dump(check_counter, fp)
    
    return modified_fisher