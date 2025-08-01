import sys,os,argparse,time
import numpy as np
from perf_utils import get_new_at_each_step, get_res_fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--my_save_path', type=str, default='')
    parser.add_argument('--rand_idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--best_lr_id', type=int, default=None)
    parser.add_argument('--best_lamb_i', type=int, default=None)
    parser.add_argument('--alpha_lamb_i', type=int, default=None)
    parser.add_argument('--lamb_down', type=float, default=None)
    parser.add_argument('--elasticity_up_mult', type=float, default=None)
    parser.add_argument('--growth', type=float, default=None)
    parser.add_argument('--tid', type=int, default=None)
    args = parser.parse_args()

    #Appr1: start at least alpha_lamb and increase until better than baseline without ANCL (what if never better than baseline?)
    
    try:
        alpha_lamb_array = np.load(args.my_save_path+'_lamb_down_array.npy')
        np.save(args.my_save_path+'_lamb_down_array.npy',np.concatenate((alpha_lamb_array,np.array([args.lamb_down]))))
        lamb_up_array = np.load(args.my_save_path+'_lamb_up_array.npy')
        np.save(args.my_save_path+'_lamb_up_array.npy',np.concatenate((lamb_up_array,np.array([args.elasticity_up_mult]))))
    except FileNotFoundError:
        np.save(args.my_save_path+'_lamb_down_array.npy',np.array([args.lamb_down]))
        np.save(args.my_save_path+'_lamb_up_array.npy',np.array([args.elasticity_up_mult]))
    
    load_path = args.my_save_path + '.' + str(args.best_lamb_i) + '.LA_phase.' + str(args.alpha_lamb_i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
    task_f1 = get_new_at_each_step(load_path)[args.tid]
    
    load_path = args.my_save_path + '.' + str(args.best_lamb_i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
    baseline_f1 = get_new_at_each_step(load_path)[args.tid]
    
    load_path = args.my_save_path + '_gold.' + str(args.best_lr_id) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
    best_f1 = get_new_at_each_step(load_path)[args.tid]
    
    l3 = []
    for i in range(1,args.alpha_lamb_i+1,1):
        load_path = args.my_save_path + '.' + str(args.best_lamb_i) + '.LA_phase.' + str(i) + '/' + get_res_fname(args.rand_idx,args.seed,args.my_save_path,args.dataset)
        l3.append(get_new_at_each_step(load_path)[args.tid])
    l3_recent = l3[-10:]
    try:
        slope = np.polyfit(range(len(l3_recent)),l3_recent,1)[0]
    except (ValueError,SystemError):
        slope = 1 # In case of error (eg. len(l3)=1), skip slope evaluation and run next lamb_down
    
    if (task_f1 > baseline_f1) or (task_f1 == best_f1) or (slope < 0.003 and len(l3) > 15 and args.elasticity_up_mult < 0.1): # (Reached good perf) or (exhausted lamb_down and lamb_up search budget)
        with open(args.my_save_path+ '.' + str(args.best_lamb_i) + '.LA_phase.' + str(args.alpha_lamb_i) + '_foundbestlambdown.txt', 'w') as file:
            file.write(str('found'))
        return # using string since shell script does not work with boolean
    else:
        if (slope < 0.003 and len(l3) > 15 and args.elasticity_up_mult == 1.0) or (args.elasticity_up_mult < 1.0): # (Exhausted lamb_down search budget) or (currently using lamb_up search budget)
            next_alpha_lamb = args.lamb_down
            next_lamb_up = args.growth * args.elasticity_up_mult
        else: # First use lamb_down search budget
            next_alpha_lamb = args.growth * args.lamb_down
            next_lamb_up = args.elasticity_up_mult
        with open(args.my_save_path+'_next_lamb_down.txt', 'w') as file:
            file.write(str(next_alpha_lamb))
        with open(args.my_save_path+'_next_lamb_up.txt', 'w') as file:
            file.write(str(next_lamb_up))
        with open(args.my_save_path+ '.' + str(args.best_lamb_i) + '.LA_phase.' + str(args.alpha_lamb_i) + '_foundbestlambdown.txt', 'w') as file:
            file.write(str('notfound'))
        return # using string since shell script does not work with boolean


if __name__ == '__main__':
    sys.exit(main())