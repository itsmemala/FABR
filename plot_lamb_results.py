import sys,os,argparse,time
import numpy as np
from matplotlib import pyplot as plt
from perf_utils import get_new_at_each_step, get_f1_at_each_step, get_forg_at_each_step, get_res_fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--my_save_path', type=str, default='')
    parser.add_argument('--rand_idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--lamb_i', type=int, default=None)
    parser.add_argument('--lamb', type=float, default=None)
    parser.add_argument('--acc_drop_threshold', type=float, default=None)
    parser.add_argument('--tid', type=int, default=None)
    args = parser.parse_args()
    
    lamb_array = np.load(args.my_save_path+'_lamb_array.npy')
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.subplots_adjust(wspace=1)
    l1, l2, l3 = [], [], []
    for i in range(1,args.lamb_i+1,1):
        load_path = args.my_save_path + '.' + str(i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
        l1.append(get_f1_at_each_step(load_path)[args.tid])
        l2.append(get_forg_at_each_step(load_path)[args.tid])
        l3.append(get_new_at_each_step(load_path)[args.tid])
        
    axes[0].plot(range(len(l1)), l1, marker='o', color='blue')
    axes[1].plot(range(len(l2)), l2, marker='o', color='blue')
    axes[2].plot(range(len(l3)), l3, marker='o', color='blue')
    
    gold_f1 = np.load(args.my_save_path+'_gold_return_best_lr_script_result.npy')[1]
    threshold_f1 = (1 - args.acc_drop_threshold) * gold_f1
    axes[2].plot(range(len(l3)), [threshold_f1 for x in range(len(l3))], linestyle='-.', color='gold')
    
    axes[0].title.set_text('Mean performance on all seen tasks')
    axes[1].title.set_text('Mean forgetting on old tasks')
    axes[2].title.set_text('New task performance')
    fig.savefig(args.my_save_path+'_lamb_results.png')

if __name__ == '__main__':
    sys.exit(main())