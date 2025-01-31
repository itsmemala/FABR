from matplotlib import pyplot as plt
import numpy as np
import re

def get_f1_at_each_step(path):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    return [np.mean(f1_matrix[i,:i+1]) for i in range(6)]

def get_forg_at_each_step(path):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    bwt = [0]
    for i in [1,2,3,4,5]:
        temp_bwt=[]
        for j in range(i):
            temp_bwt.append(f1_matrix[i,j]-f1_matrix[i-1,j])
        bwt.append(np.mean(temp_bwt))
    return bwt

def get_new_at_each_step(path):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    return [f1_matrix[i,i] for i in range(6)]

def get_overall_f1(path):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    return np.mean(f1_matrix[1,:2])

def get_overall_f1_all(path,t=6):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    if t==6:
      return np.mean(f1_matrix[5,:])
    else: # t is a list
      return np.mean([f1_matrix[5,i] for i in t])

def get_forgetting(path):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    temp_forgetting = []
    for i in range(1): # for i in range(5):
        temp_forgetting.append(np.max(f1_matrix[i:-1,i])-f1_matrix[1,i]) # temp_forgetting.append(np.max(f1_matrix[i:-1,i])-f1_matrix[5,i])
    return np.mean(temp_forgetting)

def get_forgetting_all(path,t=6):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    temp_forgetting = []
    for i in range(5):
        temp_forgetting.append(np.max(f1_matrix[i:-1,i])-f1_matrix[5,i])
    if t==6:
        return np.mean(temp_forgetting)
    else: # t is a list
        return np.mean([temp_forgetting[i] for i in t if i!=5])

def get_newtask(path):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    return f1_matrix[1,1]

def get_newtask_all(path,t=6):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    new_task = [f1_matrix[i,i] for i in range(6)]
    if t==6:
        return np.mean(new_task)
    else:
        return np.mean([new_task[i] for i in t])

def get_oldtask(path):
    list_of_lists = []
    with open(path, 'r') as f:
        for line in f:
            inner_list = [float(elt.strip()) for elt in line.split('\t')]
            list_of_lists.append(inner_list)
    f1_matrix = np.array(list_of_lists)
    return f1_matrix[1,0]

# def get_res_fname(rand_idx,seed,path,dataset):
    # if 'ANCLMAS' in path or 'ANCLEWC' in path:
        # return dataset+'_bert_adapter_ewc_ancl_'+'random'+str(rand_idx)+'_seed'+str(seed)+'_f1.txt'
    # elif 'ANCLLWF' in path:
        # return dataset+'_bert_adapter_lwf_ancl_'+'random'+str(rand_idx)+'_seed'+str(seed)+'_f1.txt'
    # elif 'LWF' in path:
        # return dataset+'_bert_adapter_lwf_'+'random'+str(rand_idx)+'_seed'+str(seed)+'_f1.txt'
    # elif 'LAEWC' in path or 'LAMAS' in path:
        # return dataset+'_bert_adapter_ewc_freeze_'+'random'+str(rand_idx)+'_seed'+str(seed)+'_f1.txt'
    # else:
        # return dataset+'_bert_adapter_ewc_'+'random'+str(rand_idx)+'_seed'+str(seed)+'_f1.txt'

def get_res_fname(rand_idx,seed,path,dataset,val=True):
    path_append = '_val.txt' if val else '.txt'
    if 'ANCLMAS' in path or 'ANCLEWC' in path:
        return dataset+'_bert_adapter_ewc_ancl_'+'random'+str(rand_idx)+'_seed'+str(seed)+'_f1'+path_append
    elif 'ANCLLWF' in path:
        return dataset+'_bert_adapter_lwf_ancl_'+'random'+str(rand_idx)+'_seed'+str(seed)+'_f1'+path_append
    elif 'LAEWC' in path or 'LAMAS' in path:
        return dataset+'_bert_adapter_ewc_freeze_'+'random'+str(rand_idx)+'_seed'+str(seed)+'_f1'+path_append