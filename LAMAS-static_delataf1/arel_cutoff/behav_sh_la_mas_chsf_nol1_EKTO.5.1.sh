#!/bin/bash

# set -Eeuo pipefail

# Initialise the following: res_path, lr_array, decay, acc_drop_threshold, growth
note=$1 #random10
randid=$2 #10
seed=$3 #0
custom_max_lamb=$4
dataset='annomi'
lr_array=(0.00003 0.0003 0.003 0.03)
decay=0.9
acc_drop_threshold=0.05
growth=0.1
res_path="/home/local/data/ms/fabr_data/BehavSH/BehavSH_LAMAS_NoL1_EKTO.5.1/${note}seed${seed}_${acc_drop_threshold}adt/BehavSH_LAMAS_t"
base_res_path="/home/local/data/ms/fabr_data/BehavSH/BehavSH_LAMAS_NoL1_EKTO.5/${note}seed${seed}_${acc_drop_threshold}adt/BehavSH_LAMAS_t"

# id=0
# printf "\n\nRunning search for task 0\n\n"
# lr_id=2
# lr_array_t0=(0.0003)
# for lr in "${lr_array_t0[@]}"
# do
	# # ((lr_id++))
	# printf "\n\nLR Iteration $lr\n\n"
	# mkdir -p  ${res_path}${id}_gold.${lr_id}/
	# CUDA_VISIBLE_DEVICES=0 python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --use_cls_wgts True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --learning_rate $lr --fisher_combine max  --break_after_task 0 --my_save_path ${res_path}${id}_gold.${lr_id}/ --only_mcl True
# done

# # python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
# # best_lr_id=$?
# best_lr_id=2
# past_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
# past_lamb=0

start_model_path="${base_res_path}0_gold.2/"

id_array=(1)
for id in "${id_array[@]}"
do
	# printf "\n\nRunning search for task $id\n\n"
	# lr_id=2
	# lr_array_t0=(0.0003)
	# for lr in "${lr_array_t0[@]}"
	# do
		# # ((lr_id++))
		# printf "\n\nLR Iteration $lr\n\n"
		# custom_lamb="$past_lamb,0"
		# custom_lr="$past_lr,$lr"
		# mkdir -p  ${res_path}${id}_gold.${lr_id}/
		# CUDA_VISIBLE_DEVICES=0 python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --use_cls_wgts True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine max  --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
	# done
	
	# # python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
	# # best_lr_id=$?
	# best_lr_id=2
	# best_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
	# past_lr="$past_lr,$best_lr"
	# python3 FABR/calc_max_lamb.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --best_lr_id $best_lr_id --best_lr $best_lr --tid $id --tid $id --custom_max_lamb $custom_max_lamb
	# start_lamb=$(<${res_path}${id}_gold_max_lamb.txt)
	# if [ "$id" -gt 1 ]; then
		# start_lamb=$best_lamb
	# fi

	# ## Lamb
	# lamb=$start_lamb
	# lamb_i=0
	# found_best=false
	# while [ $found_best=false ]
	# do
		# ((lamb_i++))
		# custom_lr=$past_lr
		# custom_lamb="$past_lamb,$lamb"
		# printf "\n\nLamb Iteration $custom_lamb \n\n"
		# mkdir -p  ${res_path}${id}.${lamb_i}/
		# CUDA_VISIBLE_DEVICES=0 python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --use_cls_wgts True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine max  --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
		# # python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
		# # found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
		# # python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
		# # if [ $found_best = found ]; then
			# # best_lamb=$lamb
			# # best_lamb_i=$lamb_i
			# # break
		# # fi
		# # lamb=`cat ${res_path}${id}_next_lamb.txt`
		# best_lamb=$lamb
		# break
	# done
	
	# past_lamb="$past_lamb,$best_lamb"
	
	past_lr=0.0003,0.0003
	past_lamb="0,$custom_max_lamb"
	
	## With LA phase
	custom_lr=$past_lr
	custom_lamb=$past_lamb
	printf "\n\nLA Phase\n\n"
	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase/
	CUDA_VISIBLE_DEVICES=0 python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --use_cls_wgts True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine max  --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase/ --start_at_task $id --start_model_path $start_model_path --adapt_type orig_enablektonly --elasticity_up 0.01 --frel_cut 0.1
		
	start_model_path="${res_path}${id}.${best_lamb_i}.LA_phase/"
done


