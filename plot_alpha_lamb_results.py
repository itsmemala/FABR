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
    parser.add_argument('--best_lamb_i', type=int, default=None)
    parser.add_argument('--alpha_lamb_i', type=int, default=None)
    parser.add_argument('--alpha_lamb', type=float, default=None)
    parser.add_argument('--tid', type=int, default=None)
    args = parser.parse_args()
    
    alpha_lamb_array = np.load(args.my_save_path+'_alpha_lamb_array.npy')
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    l1, l2, l3 = [], [], []
    for i in range(1,args.alpha_lamb_i+1,1):
        load_path = args.my_save_path + '.' + str(best_lamb_i) + '.' + str(i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
        l1.append(get_f1_at_each_step(load_path)[args.tid])
        l2.append(get_forg_at_each_step(load_path)[args.tid])
        l3.append(get_new_at_each_step(load_path)[args.tid])
        
    axes[0,0].plot(range(len(l1)), l1, marker='o', color='blue')
    axes[0,1].plot(range(len(l2)), l2, marker='o', color='blue')
    axes[0,2].plot(range(len(l3)), l3, marker='o', color='blue')
    
    load_path = args.my_save_path + '.' + str(args.best_lamb_i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
    baseline_f1 = get_new_at_each_step(load_path)[args.tid]
    axes[0,2].plot(range(len(l3)), [baseline_f1 for x in range(len(l3))], linestyle='-.', color='gold')
    
    axes[0,0].title.set_text('Mean performance on all seen tasks')
    axes[0,1].title.set_text('Mean forgetting on old tasks')
    axes[0,2].title.set_text('New task performance')
    fig.savefig(args.my_save_path+'/'+'alpha_lamb_results.png')

if __name__ == '__main__':
    sys.exit(main())