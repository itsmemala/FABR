import sys,os,argparse,time
import numpy as np
from perf_utils import get_new_at_each_step, get_res_fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--my_save_path', type=str, default='')
    parser.add_argument('--rand_idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--best_lamb_i', type=int, default=None)
    parser.add_argument('--alpha_lamb_i', type=int, default=None)
    parser.add_argument('--alpha_lamb', type=float, default=None)
    parser.add_argument('--growth', type=float, default=None)
    parser.add_argument('--tid', type=int, default=None)
    args = parser.parse_args()

    #Appr1: start at least alpha_lamb and increase until better than baseline without ANCL (what if never better than baseline?)
    
    try:
        alpha_lamb_array = np.load(args.my_save_path+'_alpha_lamb_array.npy')
        np.save(args.my_save_path+'_alpha_lamb_array.npy',np.concatenate((alpha_lamb_array,args.alpha_lamb)))
    except FileNotFoundError:
        np.save(args.my_save_path+'_alpha_lamb_array.npy',np.array([args.alpha_lamb]))
    
    load_path = args.my_save_path + '.' + str(args.best_lamb_i) + '.' + str(args.alpha_lamb_i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
    task_f1 = get_new_at_each_step(load_path)[args.tid]
    
    load_path = args.my_save_path + '.' + str(args.best_lamb_i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
    baseline_f1 = get_new_at_each_step(load_path)[args.tid]
    
    if task_f1 > baseline_f1:
        return 'true' # using string since shell script does not work with boolean
    else:
        next_alpha_lamb = (1 + args.growth) * args.alpha_lamb
        with open(args.my_save_path+'_next_alpha_lamb.txt', 'wb') as file:
            file.write(next_alpha_lamb)
        return 'false' # using string since shell script does not work with boolean


if __name__ == '__main__':
    sys.exit(main())