#!/bin/bash

# Initialise the following: lr, decay, acc_drop_threshold

past_lamb = 0
past_alpha_lamb = 0

for id in 1 2 3 4 5
do

	lr_id = 0
	for lr in 0.00003, 0.0003, 0.003, 0.03
	do
		((lr_id++))
		echo "LR Iteration $lr"
		CUDA_VISIBLE_DEVICES=0 python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_ancl --imp function --baseline ewc_ancl --backbone bert_adapter --note random0 --idrandom 0 --seed 0 --scenario dil --use_cls_wgts True --train_batch_size 32 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --learning_rate $lr --remove_lr_schedule True --remove_wd True --custom_lamb 0,0 --custom_alpha_lamb 0,0 --ancl True --break_after_task $id --my_save_path ~/fabr_data/BehavSH/BehavSH_ANCLMAS_t${id}_gold.${lr_id}/
	done
	task_gold, best_lr = #TODO

	echo "Task $id"
	## Lamb
	lamb, lamb_i = 0.1, 0
	found_best = False
	while [ found_best = False ]
	do
		((lamb_i++))
		custom_lamb = $past_lamb,$lamb
		custom_alpha_lamb = $past_alpha_lamb,0
		echo "Iteration $custom_lamb $custom_alpha_lamb"
		CUDA_VISIBLE_DEVICES=0 python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_ancl --imp function --baseline ewc_ancl --backbone bert_adapter --note random0 --idrandom 0 --seed 0 --scenario dil --use_cls_wgts True --train_batch_size 32 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --learning_rate $best_lr --remove_lr_schedule True --remove_wd True --custom_lamb $custom_lamb --custom_alpha_lamb $custom_alpha_lamb --ancl True --break_after_task $id --my_save_path ~/fabr_data/BehavSH/BehavSH_ANCLMAS_t${id}.${lamb_i}/
		found_best = #TODO
		best_lamb, best_lamb_i = #TODO
		lamb = #TODO
	done
	
	past_lamb = $past_lamb,$best_lamb
	
	## Alpha lamb
	alpha_lamb, alpha_lamb_i = 0.1, 0
	found_best = False
	while [ found_best = False ]
	do
		((alpha_lamb_i++))
		custom_lamb = $past_lamb
		custom_alpha_lamb = $past_alpha_lamb,$alpha_lamb
		echo "Iteration $custom_lamb $custom_alpha_lamb"
		CUDA_VISIBLE_DEVICES=0 python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_ancl --imp function --baseline ewc_ancl --backbone bert_adapter --note random0 --idrandom 0 --seed 0 --scenario dil --use_cls_wgts True --train_batch_size 32 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --learning_rate $best_lr --remove_lr_schedule True --remove_wd True --custom_lamb $custom_lamb --custom_alpha_lamb $custom_alpha_lamb --ancl True --break_after_task $id --my_save_path ~/fabr_data/BehavSH/BehavSH_ANCLMAS_t${id}.${best_lamb_i}.${alpha_lamb_i}/
		found_best = #TODO
		best_alpha_lamb, best_alpha_lamb_i = #TODO
		alpha_lamb = #TODO
	done
	
	past_alpha_lamb = $past_alpha_lamb,$best_alpha_lamb
done


