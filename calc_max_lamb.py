import sys,os,argparse,time
import numpy as np
from utils import CPU_Unpickler
from perf_utils import get_new_at_each_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--my_save_path', type=str, default='')
    parser.add_argument('--rand_idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--best_lr_id', type=int, default=None)
    parser.add_argument('--best_lr', type=float, default=None)
    parser.add_argument('--tid', type=int, default=None)
    parser.add_argument('--custom_max_lamb', type=float, default=None) # Override with this value if provided
    args = parser.parse_args()
    
    load_path = args.my_save_path + '.' + str(args.best_lr_id) + '/'
    with open(load_path+'fisher_old.pkl', 'rb') as handle:
        alpha_rel = CPU_Unpickler(handle).load()
    
    vals = np.array([])
    for k,v in alpha_rel.items():
        vals = np.append(vals,v.flatten().numpy())
    
    if args.custom_max_lamb is not None:
        max_lamb = args.custom_max_lamb
    else:
        max_lamb = 1/(args.best_lr * np.max(vals)) # lambda < 1/(eta * alpha)
    
    # write to file
    with open(args.my_save_path+'_max_lamb.txt', 'w') as file:
        file.write(str(max_lamb))
    
    return
    
if __name__ == '__main__':
    sys.exit(main())