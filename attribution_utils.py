import numpy as np
import statistics
import torch

def aggregate_local_to_global(attributions,predictions,targets,tokens):
    print("Aggregating attributions from local to global")
    print('*'*10)
    
    all_tokens = []
    for example in tokens:
            all_tokens += example
    all_tokens = list(set(all_tokens))
    print("Total unique features:", len(all_tokens))
    
    mxtx_tp = np.where((predictions==0) & (predictions==targets))[0]
    mxtx_tn = np.where((predictions==1) & (predictions==targets))[0]

    attr_pos_mxtx = np.clip(attributions, a_min=0, a_max=None)
    attr_neg_mxtx = np.clip(attributions, a_min=None, a_max=0)
    
    check1 = 90
    check2 = 10

    # Collect attributions of each feature across samples
    # true positives
    attr_mxtx_tp = attr_pos_mxtx[mxtx_tp]
    mxtx_pos_token_global_attr = {}
    for train_idx,attr in zip(mxtx_tp,attr_mxtx_tp):
        train_tokens = tokens[train_idx]
        token_attr = attr[1:len(train_tokens)+1]
        check_attr1=np.percentile(token_attr,check1)
        check_attr2=np.percentile(token_attr,check2)
        for token,attr_val in zip(train_tokens,token_attr):
            attr_val = 1 if attr_val>check_attr1 else 0
            if token in mxtx_pos_token_global_attr:
                mxtx_pos_token_global_attr[token].append(attr_val)
            else:
                mxtx_pos_token_global_attr[token] = [attr_val]
    attr_mxtx_tp = attr_neg_mxtx[mxtx_tp]
    mxtx_neg_token_global_attr = {}
    for train_idx,attr in zip(mxtx_tp,attr_mxtx_tp):
        train_tokens = tokens[train_idx]
        token_attr = attr[1:len(train_tokens)+1]
        check_attr1=np.percentile(token_attr,check1)
        check_attr2=np.percentile(token_attr,check2)
        for token,attr_val in zip(train_tokens,token_attr):
            attr_val = -1 if attr_val<check_attr2 else 0
            if token in mxtx_neg_token_global_attr:
                mxtx_neg_token_global_attr[token].append(attr_val*-1)
            else:
                mxtx_neg_token_global_attr[token] = [attr_val*-1]
    # true negatives
    attr_mxtx_tn = attr_neg_mxtx[mxtx_tn]
    for train_idx,attr in zip(mxtx_tn,attr_mxtx_tn):
        train_tokens = tokens[train_idx]
        token_attr = attr[1:len(train_tokens)+1]
        check_attr1=np.percentile(token_attr,check1)
        check_attr2=np.percentile(token_attr,check2)
        for token,attr_val in zip(train_tokens,token_attr):
            attr_val = -1 if attr_val<check_attr2 else 0
            if token in mxtx_pos_token_global_attr:
                mxtx_pos_token_global_attr[token].append(attr_val*-1)
            else:
                mxtx_pos_token_global_attr[token] = [attr_val*-1]
    attr_mxtx_tn = attr_pos_mxtx[mxtx_tn]
    for train_idx,attr in zip(mxtx_tn,attr_mxtx_tn):
        train_tokens = tokens[train_idx]
        token_attr = attr[1:len(train_tokens)+1]
        check_attr1=np.percentile(token_attr,check1)
        check_attr2=np.percentile(token_attr,check2)
        for token,attr_val in zip(train_tokens,token_attr):
            attr_val = 1 if attr_val>check_attr1 else 0
            if token in mxtx_neg_token_global_attr:
                mxtx_neg_token_global_attr[token].append(attr_val)
            else:
                mxtx_neg_token_global_attr[token] = [attr_val]
    # false positives
    attr_mxtx_fp = attr_pos_mxtx[mxtx_fp]
    for train_idx,attr in zip(mxtx_fp,attr_mxtx_fp):
        train_tokens = tokens[str(j)][train_idx]
        token_attr = attr[1:len(train_tokens)+1]
        check_attr1=np.percentile(token_attr,check1)
        check_attr2=np.percentile(token_attr,check2)
        for token,attr_val in zip(train_tokens,token_attr):
            attr_val = 1 if attr_val>check_attr1 else 0
            if token in mxtx_pos_token_global_attr:
                mxtx_pos_token_global_attr[token].append(attr_val)
            else:
                mxtx_pos_token_global_attr[token] = [attr_val]
    attr_mxtx_fp = attr_neg_mxtx[mxtx_fp]
    for train_idx,attr in zip(mxtx_fp,attr_mxtx_fp):
        train_tokens = tokens[str(j)][train_idx]
        token_attr = attr[1:len(train_tokens)+1]
        check_attr1=np.percentile(token_attr,check1)
        check_attr2=np.percentile(token_attr,check2)
        for token,attr_val in zip(train_tokens,token_attr):
            attr_val = -1 if attr_val<check_attr2 else 0
            if token in mxtx_neg_token_global_attr:
                mxtx_neg_token_global_attr[token].append(attr_val*-1)
            else:
                mxtx_neg_token_global_attr[token] = [attr_val*-1]
    # false negatives
    attr_mxtx_fn = attr_neg_mxtx[mxtx_fn]
    for train_idx,attr in zip(mxtx_fn,attr_mxtx_fn):
        train_tokens = tokens[str(j)][train_idx]
        token_attr = attr[1:len(train_tokens)+1]
        check_attr1=np.percentile(token_attr,check1)
        check_attr2=np.percentile(token_attr,check2)
        for token,attr_val in zip(train_tokens,token_attr):
            attr_val = -1 if attr_val<check_attr2 else 0
            if token in mxtx_pos_token_global_attr:
                mxtx_pos_token_global_attr[token].append(attr_val*-1)
            else:
                mxtx_pos_token_global_attr[token] = [attr_val*-1]
    attr_mxtx_fn = attr_pos_mxtx[mxtx_fn]
    for train_idx,attr in zip(mxtx_fn,attr_mxtx_fn):
        train_tokens = tokens[str(j)][train_idx]
        token_attr = attr[1:len(train_tokens)+1]
        check_attr1=np.percentile(token_attr,check1)
        check_attr2=np.percentile(token_attr,check2)
        for token,attr_val in zip(train_tokens,token_attr):
            attr_val = 1 if attr_val>check_attr1 else 0
            if token in mxtx_neg_token_global_attr:
                mxtx_neg_token_global_attr[token].append(attr_val)
            else:
                mxtx_neg_token_global_attr[token] = [attr_val]

    # Aggregate attributions of each feature to global attributions
    mxtx_pos_mean_global_attr_dict = {}
    pos_cnt=0
    mxtx_neg_mean_global_attr_dict = {}
    neg_cnt=0
    for token in mxtx_pos_token_global_attr.keys():
        token_mean = statistics.mean(mxtx_pos_token_global_attr[token])
        mxtx_pos_mean_global_attr_dict[token] = token_mean
        if token_mean>0:
            pos_cnt += 1
    for token in mxtx_neg_token_global_attr.keys():
        token_mean = statistics.mean(mxtx_neg_token_global_attr[token])
        mxtx_neg_mean_global_attr_dict[token] = token_mean
        if token_mean>0:
            neg_cnt += 1
    
    print("Top 10% features for class 0:",pos_cnt)
    print("Top 10% features for class 1:",neg_cnt)
    
    global_attr = {}
    global_attr['pos'] = mxtx_pos_mean_global_attr_dict
    global_attr['neg'] = mxtx_neg_mean_global_attr_dict
    
    return global_attr

def get_batch_targets(attributions,classes,batch_tokens,global_attr):
    targets = []
    
    # Note: Assumes the function is called only for samples with true predictions, using pred class attributions
    # Or for all samples, using actual class attributions
    for attr,cls,example in zip(attributions,classes,batch_tokens): # loop through each example
        if cls==0:
            global_attr_pred = global_attr['pos']
            global_attr_opp = global_attr['neg']
        else:
            global_attr_pred = global_attr['neg']
            global_attr_opp = global_attr['pos']
        
        example_target = []
        for token in example: # loop through each token and set an attribution target
            if token in global_attr_pred:
                # Regularize if the feature had high attribution previously towards predicted class(to prevent forgetting)
                if global_attr_pred[token]>=0.5:
                    target_attr =  1
                # Else, regularize in opposite direction if it had high attribution towards opposite class previously (to prevent forgetting)
                elif token in global_attr_opp:
                    if global_attr_opp[token]>=0.5:
                        target_attr =  -1
                # Otherwise, do not regularize (to allow learning)
                    else:
                        target_attr = attr
                else:
                    target_attr = attr
            elif token in global_attr_pred:
                # Else, regularize in opposite direction if it had high attribution towards opposite class previously (to prevent forgetting)
                if global_attr_pred[token]>=0.5:
                    target_attr =  1
                # Otherwise, do not regularize (to allow learning)
                else:
                    target_attr = attr
            else:
                # Do not regularize if it's a new feature not seen in previous tasks (to allow learning)
                # Note: Since we do not calculate global attr for cls, sep and pad tokens, attributions for those tokens will not be regularized
                # Does this work as intended during backprop?
                target_attr = attr
            
            example_target.append(target_attr)
        targets.append(example_target)
    
    targets = torch.Tensor(targets)
    assert targets.shape==attributions.shape
    
    return targets

