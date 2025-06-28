#!/bin/bash

# set -Eeuo pipefail

# Initialise the following: res_path, lr_array, decay, acc_drop_threshold, growth
note=$1 #random10
randid=$2 #10
seed=$3 #0
svd_thres=$4
dataset='hwu64'
lr_array_t0=(0.00003 0.0003)
# lr_array=(0.00003 0.00003 0.00003 0.00003 0.0003 0.0003 0.0003 0.0003)
# tc_epsilon_array=(0.3 0.3 0.5 0.5 0.3 0.3 0.5 0.5)
# tc_lamb_s_array=(0.3 0.08 0.3 0.08 0.3 0.08 0.3 0.08)
# tc_lamb_l_array=(0.1 0.03 0.1 0.03 0.1 0.03 0.1 0.03)
lr_array=(0.00003 0.00003 0.00003 0.00003 0.0003 0.0003 0.0003 0.0003)
tc_epsilon_array=(0.3 0.3 0.5 0.5 0.3 0.3 0.5 0.5)
tc_lamb_s_array=(0.3 0.08 0.3 0.08 0.3 0.08 0.3 0.08)
tc_lamb_l_array=(0.1 0.03 0.1 0.03 0.1 0.03 0.1 0.03)
res_path="/home/local/data/ms/fabr_data/IntentSH/Adabop_svd${svd_thres}/${note}seed${seed}/IntentSH_Adabop_t"

id=0
printf "\n\nRunning search for task 0\n\n"
lr_id=0
for lr in "${lr_array_t0[@]}"
do
	((lr_id++))
	printf "\n\nLR,LAMB Iteration $lr\n\n"
	mkdir -p  ${res_path}${id}_gold.${lr_id}/
	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_adabop --backbone bert_adapter --baseline adabop --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 30 --eval_batch_size 128 --valid_loss_es 0.002 --lr_patience 3 --learning_rate $lr --break_after_task 0 --my_save_path ${res_path}${id}_gold.${lr_id}/
done

python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
best_lr_id=$?
past_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing

start_model_path="${res_path}${id}_gold.${best_lr_id}/"

id_array=(1 2 3 4 5)
for id in "${id_array[@]}"
do
	printf "\n\nRunning search for task $id\n\n"
	lr_id=0
	for lr in "${lr_array[@]}"
	do
		tc_epsilon=${tc_epsilon_array[$lr_id]}
		tc_lamb_s=${tc_lamb_s_array[$lr_id]}
		tc_lamb_l=${tc_lamb_l_array[$lr_id]}
		((lr_id++))
		printf "\n\nLR Iteration $lr\n\n"
		custom_lr="$past_lr,$lr"
		mkdir -p  ${res_path}${id}_gold.${lr_id}/
		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_adabop --backbone bert_adapter --baseline adabop --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 30 --eval_batch_size 128 --valid_loss_es 0.002 --lr_patience 3 --custom_lr $custom_lr --tc_epsilon $tc_epsilon --tc_lamb_s $tc_lamb_s --tc_lamb_l $tc_lamb_l --break_after_task $id --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --svd_thres $svd_thres
	done
	
	python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
	best_lr_id=$?
	best_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
	past_lr="$past_lr,$best_lr"
		
	start_model_path="${res_path}${id}_gold.${best_lr_id}/"
done


