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
    args = parser.parse_args()
    
    load_path = args.my_save_path + '.' + str(args.best_lr_id) + '/'
    with open(load_path+'fisher_old.pkl', 'rb') as handle:
        alpha_rel = CPU_Unpickler(handle).load()
    
    vals = np.array([])
    for k,v in alpha_rel.items():
        vals = np.append(vals,v.flatten().numpy())
    
    max_lamb = 1/(args.best_lr * np.max(vals)) # lambda < 1/(eta * alpha)
    
    # write to file
    with open(args.my_save_path+'_max_lamb.txt', 'wb') as file:
        file.write(max_lamb)
    
    return
    
if __name__ == '__main__':
    sys.exit(main())