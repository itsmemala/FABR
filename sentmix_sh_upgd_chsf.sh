#!/bin/bash

# set -Eeuo pipefail

# Initialise the following: res_path, lr_array, decay, acc_drop_threshold, growth
note=$1 #random10
randid=$2 #10
seed=$3 #0
choose_second_best=$4
dataset='sent_mix'
lr_array=(0.00003) #(0.00003 0.0003 0.003 0.03)
res_path="/home/local/data/ms/fabr_data/SentMixSH/UPGD_FixedLR_CSB${choose_second_best}/${note}seed${seed}/SentMixSH_UPGD_t"

id=0
printf "\n\nRunning search for task 0\n\n"
lr_id=0
for lr in "${lr_array[@]}"
do
	((lr_id++))
	printf "\n\nLR Iteration $lr\n\n"
	mkdir -p  ${res_path}${id}_gold.${lr_id}/
	python FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_upgd --backbone bert_adapter --baseline upgd --note $note --idrandom $randid --seed $seed --scenario dil --use_cls_wgts True --train_batch_size 128 --num_train_epochs 50 --eval_batch_size 128 --valid_loss_es 0.002 --lr_patience 5 --learning_rate $lr --break_after_task 0 --my_save_path ${res_path}${id}_gold.${lr_id}/
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
		((lr_id++))
		printf "\n\nLR Iteration $lr\n\n"
		custom_lr="$past_lr,$lr"
		mkdir -p  ${res_path}${id}_gold.${lr_id}/
		python FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_upgd --backbone bert_adapter --baseline upgd --note $note --idrandom $randid --seed $seed --scenario dil --use_cls_wgts True --train_batch_size 128 --num_train_epochs 50 --eval_batch_size 128 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --break_after_task $id --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path
	done
	
	python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id --choose_second_best $choose_second_best
	best_lr_id=$?
	best_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
	past_lr="$past_lr,$best_lr"
		
	start_model_path="${res_path}${id}_gold.${best_lr_id}/"
done


