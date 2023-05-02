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
# v2.3
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