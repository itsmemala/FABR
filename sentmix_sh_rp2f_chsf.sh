#!/bin/bash

# set -Eeuo pipefail

# Initialise the following: res_path, lr_array, decay, acc_drop_threshold, growth
note=$1 #random10
randid=$2 #10
seed=$3 #0
learner_is_seq=$4
dataset='sent_mix'
lr_array=(0.00003 0.00003 0.00003 0.0003 0.0003 0.0003) #0.003 0.003 0.003)
lamb_array=(0.0000001 0.000001 0.00001 0.0000001 0.000001 0.00001) #0.0000001 0.000001 0.00001)
res_path="/home/local/data/ms/fabr_data/SentMixSH/RP2F_${learner_is_seq}/${note}seed${seed}/SentMixSH_RP2F_t"

id=0
printf "\n\nRunning search for task 0\n\n"
lr_id=0
for lr in "${lr_array[@]}"
do
	lamb=${lamb_array[$lr_id]}
	((lr_id++))
	printf "\n\nLR,LAMB Iteration $lr\n\n"
	mkdir -p  ${res_path}${id}_gold.${lr_id}/
	python FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_rp2f_sh --backbone bert_adapter --baseline rp2f_sh --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 30 --eval_batch_size 128 --valid_loss_es 0.002 --lr_patience 3 --learning_rate $lr --rp2f_lamb $lamb --break_after_task 0 --my_save_path ${res_path}${id}_gold.${lr_id}/ --learner_is_seq $learner_is_seq
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
		lamb=${lamb_array[$lr_id]}
		((lr_id++))
		printf "\n\nLR Iteration $lr\n\n"
		custom_lr="$past_lr,$lr"
		mkdir -p  ${res_path}${id}_gold.${lr_id}/
		python FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_rp2f_sh --backbone bert_adapter --baseline rp2f_sh --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 30 --eval_batch_size 128 --valid_loss_es 0.002 --lr_patience 3 --custom_lr $custom_lr --rp2f_lamb $lamb --break_after_task $id --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --learner_is_seq $learner_is_seq
	done
	
	python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
	best_lr_id=$?
	best_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
	past_lr="$past_lr,$best_lr"
		
	start_model_path="${res_path}${id}_gold.${best_lr_id}/"
done

###########
# CUDA_VISIBLE_DEVICES=1 bash sentmix_sh_rp2f_chsf.sh random0 0 0 True
# CUDA_VISIBLE_DEVICES=1 bash sentmix_sh_rp2f_chsf.sh random1 1 0 True
# CUDA_VISIBLE_DEVICES=1 bash sentmix_sh_rp2f_chsf.sh random2 2 0 True