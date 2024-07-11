import sys,os,argparse,time
import numpy as np
from perf_utils import get_new_at_each_step, get_res_fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--my_save_path', type=str, default='')
    parser.add_argument('--rand_idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lamb_i', type=int, default=None)
    parser.add_argument('--lamb', type=float, default=None)
    parser.add_argument('--decay', type=float, default=None)
    parser.add_argument('--acc_drop_threshold', type=float, default=None)
    parser.add_argument('--tid', type=int, default=None)
    args = parser.parse_args()

    #Appr1: start at max lamb and decrease until within acc_drop_threshold (how to set this? per task? what if never within threshold?)
    #Appr2: start at high lamb (how to set this?) and decrease until within acc_drop_threshold (how to set this? per task? what if never within threshold?)
    #Appr3: start at prev task best lamb and decrease until within acc_drop_threshold (how to set this? per task? what if never within threshold?)
    
    try:
        lamb_array = np.load(args.my_save_path+'_lamb_array.npy')
        np.save(args.my_save_path+'_lamb_array.npy',np.concatenate((lamb_array,args.lamb)))
    except FileNotFoundError:
        np.save(args.my_save_path+'_lamb_array.npy',np.array([args.lamb]))
      
    load_path = args.my_save_path + '.' + str(args.lamb_i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
    task_f1 = get_new_at_each_step(load_path)[args.tid]
    
    gold_f1 = np.load(args.my_save_path+'_return_best_lr_script_result.npy')[1]
    
    if task_f1 >= ((1 - args.acc_drop_threshold) * gold_f1)
        return 'true' # using string since shell script does not work with boolean
    else:
        next_lamb = args.decay * args.lamb
        with open(args.my_save_path+'_next_lamb.txt', 'wb') as file:
            file.write(next_lamb)
        return 'false' # using string since shell script does not work with boolean


if __name__ == '__main__':
    sys.exit(main())