#!/bin/bash

# set -Eeuo pipefail

# Initialise the following: res_path, lr_array, decay, acc_drop_threshold, growth
note=$1 #random10
randid=$2 #10
seed=$3 #0
custom_max_lamb=$4
elasticity_up_max_lamb=$5
elasticity_up_mult=$6
lamb_down=$7
pdm_frac=$8
no_frel_cut_max=$9
dataset='annomi'
lr_array=(0.00003 0.0003) #(0.00003 0.0003 0.003 0.03)
decay=0.9
acc_drop_threshold=${10}
growth=0.8
res_path="/home/local/data/ms/fabr_data/BehavSH/BehavSH_LAMAS_NoL1_Custom_cus_${pdm_frac}pdmfrac${no_frel_cut_max}/${note}seed${seed}_${acc_drop_threshold}adt/BehavSH_LAMAS_t"
# start_res_path="/home/local/data/ms/fabr_data/BehavSH/BehavSH_LAMAS_NoL1_Custom_ss_${pdm_frac}pdmfrac${no_frel_cut_max}/${note}seed${seed}_${acc_drop_threshold}adt/BehavSH_LAMAS_t"

past_lr=0.00003,0.00003,0.0003,0.00003,0.00003,0.00003
past_lamb=0,250.15772549,0.0358578885,0.00000571101,0.00000571101,1.00 #0.00000571101
# best_lamb=
# start_model_path="${start_res_path}4..LA_phase./"
start_model_path="${res_path}4.1.LA_phase.27/"

id_array=(5)
for id in "${id_array[@]}"
do
	best_lr_id=1
	best_lamb=1.0 #0.00000571101
	best_lamb_i=1 #1
	
	la_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.1/"
	
	## Lamb Down
	lamb_down=0.5
	elasticity_up_mult=0.8
	alpha_lamb_i=4
	found_best=false
	while [ $found_best=false ]
	do
		((alpha_lamb_i++))
		custom_lr=$past_lr
		custom_lamb=$past_lamb
		printf "\n\nLA Phase\n\n"
		mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --use_cls_wgts True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine max --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path
		python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
		found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
		python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
		if [ $found_best = found ]; then
			best_alpha_lamb_i=$alpha_lamb_i
			break
		fi
		lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
		elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
	done
		
	start_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.${best_alpha_lamb_i}/"
done