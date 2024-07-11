import sys,os,argparse,time
import numpy as np
from perf_utils import get_new_at_each_step, get_res_fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--my_save_path', type=str, default='')
    parser.add_argument('--rand_idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--max_lr_id', type=int, default='')
    parser.add_argument('--tid', type=int, default=None)
    args = parser.parse_args()
    
    best_lr_id, best_perf = 1, 0
    for lr_id in range(1,args.max_lr_id+1):
        load_path = args.my_save_path + '.' + str(lr_id) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
        # print(load_path)
        task_f1 = get_new_at_each_step(load_path)[args.tid]
        if task_f1 > best_perf:
            best_perf = task_f1
            best_lr_id = lr_id
    
    # write to file
    np.save(args.my_save_path+'_return_best_lr_script_result.npy', np.array([best_lr_id,best_perf]))
    
    return best_lr_id
    
if __name__ == '__main__':
    sys.exit(main())