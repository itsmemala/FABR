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
dataset='sent_mix'
lr_array=(0.00003 0.0003) # 0.003 0.03)
decay=0.9
acc_drop_threshold=${10}
growth=0.9
res_path="/home/local/data/ms/fabr_data/SentMixSH/SentMixSH_LAMAS_NoL1_Custom_ssFixedLrP20_${pdm_frac}pdmfrac${no_frel_cut_max}_AvgPoolNoLAReg/${note}seed${seed}_${acc_drop_threshold}adt/SentMixSH_LAMAS_t"

# id=0
# printf "\n\nRunning search for task 0\n\n"
# lr_id=0
# cur_lr_array=(0.00003)
# for lr in "${cur_lr_array[@]}"
# do
# 	((lr_id++))
# 	printf "\n\nLR Iteration $lr\n\n"
# 	mkdir -p  ${res_path}${id}_gold.${lr_id}/
# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --learning_rate $lr --fisher_combine avg --break_after_task 0 --my_save_path ${res_path}${id}_gold.${lr_id}/ --only_mcl True
# done

# python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
# best_lr_id=$?
# past_lr=${cur_lr_array[$best_lr_id-1]}  # -1 for array indexing
# past_lamb=0

past_lr=0.00003,0.00003,0.00003,0.00003,0.00003
past_lamb=0,500,500,75.04731765,17.1684191
best_lamb=17.1684191
# start_model_path="${res_path}1.1.LA_phase.1/"
start_model_path="${res_path}2.1.LA_phase.7/"
start_model_path="${res_path}3.19.LA_phase.9/" #select LA_phase.10?

# id_array=(1)
# for id in "${id_array[@]}"
# do
# 	# printf "\n\nRunning search for task $id\n\n"
# 	# lr_id=0
# 	# cur_lr_array=(0.00003)
# 	# for lr in "${cur_lr_array[@]}"
# 	# do
# 	# 	((lr_id++))
# 	# 	printf "\n\nLR Iteration $lr\n\n"
# 	# 	custom_lamb="$past_lamb,0"
# 	# 	custom_lr="$past_lr,$lr"
# 	# 	mkdir -p  ${res_path}${id}_gold.${lr_id}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# done
	
# 	# python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
# 	# best_lr_id=$?
# 	# best_lr=${cur_lr_array[$best_lr_id-1]}  # -1 for array indexing
# 	# past_lr="$past_lr,$best_lr"
# 	# python3 FABR/calc_max_lamb.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --best_lr_id $best_lr_id --best_lr $best_lr --tid $id --tid $id --custom_max_lamb $custom_max_lamb
# 	# start_lamb=$(<${res_path}${id}_gold_max_lamb.txt)
# 	# if [ "$id" -gt 1 ]; then
# 	# 	start_lamb=$best_lamb
# 	# fi

# 	best_lr_id=1

# 	## Lamb
# 	lamb=500
# 	lamb_i=0
# 	found_best=false
# 	while [ $found_best=false ]
# 	do
# 		((lamb_i++))
# 		custom_lr=$past_lr
# 		custom_lamb="$past_lamb,$lamb"
# 		printf "\n\nLamb Iteration $custom_lamb \n\n"
# 		mkdir -p  ${res_path}${id}.${lamb_i}/
# 		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 		python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 		found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 		python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 		if [ $found_best = found ]; then
# 			best_lamb=$lamb
# 			best_lamb_i=$lamb_i
# 			break
# 		fi
# 		lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	done
	
# 	past_lamb="$past_lamb,$best_lamb"
	
# 	# if [ "$id" -eq 1 ]; then
# 		# elasticity_up_max_lamb=`cat ${res_path}${id}_min_lamb_w_newtask_zero.txt`
# 	# fi
	
# 	la_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.1/"
	
# 	## Lamb Down
# 	lamb_down=1.0
# 	elasticity_up_mult=1.0
# 	alpha_lamb_i=0
# 	found_best=false
# 	while [ $found_best=false ]
# 	do
# 		((alpha_lamb_i++))
# 		custom_lr=$past_lr
# 		custom_lamb=$past_lamb
# 		printf "\n\nLA Phase\n\n"
# 		mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 		python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 		found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 		python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 		if [ $found_best = found ]; then
# 			best_alpha_lamb_i=$alpha_lamb_i
# 			break
# 		fi
# 		lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 		elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	done
		
# 	start_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.${best_alpha_lamb_i}/"
# done

# id_array=(2)
# for id in "${id_array[@]}"
# do
# 	# printf "\n\nRunning search for task $id\n\n"
# 	# lr_id=0
# 	# for lr in "${lr_array[@]}"
# 	# do
# 	# 	((lr_id++))
# 	# 	printf "\n\nLR Iteration $lr\n\n"
# 	# 	custom_lamb="$past_lamb,0"
# 	# 	custom_lr="$past_lr,$lr"
# 	# 	mkdir -p  ${res_path}${id}_gold.${lr_id}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# done
	
# 	# python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
# 	# best_lr_id=$?
# 	# best_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
# 	# past_lr="$past_lr,$best_lr"
# 	# python3 FABR/calc_max_lamb.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --best_lr_id $best_lr_id --best_lr $best_lr --tid $id --tid $id --custom_max_lamb $custom_max_lamb
# 	# start_lamb=$(<${res_path}${id}_gold_max_lamb.txt)
# 	# if [ "$id" -gt 1 ]; then
# 	# 	start_lamb=$best_lamb
# 	# fi

# 	# ## Lamb
# 	# lamb=$start_lamb
# 	# lamb_i=0
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	if [ $found_best = found ]; then
# 	# 		best_lamb=$lamb
# 	# 		best_lamb_i=$lamb_i
# 	# 		break
# 	# 	fi
# 	# 	lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# done
	
# 	# past_lamb="$past_lamb,$best_lamb"
	
# 	# if [ "$id" -eq 1 ]; then
# 		# elasticity_up_max_lamb=`cat ${res_path}${id}_min_lamb_w_newtask_zero.txt`
# 	# fi

# 	best_lr_id=1
# 	best_lamb_i=1
# 	best_lamb=500
	
# 	la_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.1/"
	
# 	## Lamb Down
# 	lamb_down=0.4
# 	elasticity_up_mult=1.0
# 	alpha_lamb_i=6
# 	found_best=false
# 	while [ $found_best=false ]
# 	do
# 		((alpha_lamb_i++))
# 		custom_lr=$past_lr
# 		custom_lamb=$past_lamb
# 		printf "\n\nLA Phase\n\n"
# 		mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 		python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 		found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 		python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 		# if [ $found_best = found ]; then
# 		# 	best_alpha_lamb_i=$alpha_lamb_i
# 		# 	break
# 		# fi
# 		# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 		# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 		break
# 	done
		
# 	start_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.${best_alpha_lamb_i}/"
# done

# id_array=(3)
# for id in "${id_array[@]}"
# do
# 	# printf "\n\nRunning search for task $id\n\n"
# 	# lr_id=0
# 	# for lr in "${lr_array[@]}"
# 	# do
# 	# 	((lr_id++))
# 	# 	printf "\n\nLR Iteration $lr\n\n"
# 	# 	custom_lamb="$past_lamb,0"
# 	# 	custom_lr="$past_lr,$lr"
# 	# 	mkdir -p  ${res_path}${id}_gold.${lr_id}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# done
	
# 	# python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
# 	# best_lr_id=$?
# 	# best_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
# 	# past_lr="$past_lr,$best_lr"
# 	# python3 FABR/calc_max_lamb.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --best_lr_id $best_lr_id --best_lr $best_lr --tid $id --tid $id --custom_max_lamb $custom_max_lamb
# 	# start_lamb=$(<${res_path}${id}_gold_max_lamb.txt)
# 	# if [ "$id" -gt 1 ]; then
# 	# 	start_lamb=$best_lamb
# 	# fi

# 	# ## Lamb
# 	# lamb=$start_lamb
# 	# lamb_i=0
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	if [ $found_best = found ]; then
# 	# 		best_lamb=$lamb
# 	# 		best_lamb_i=$lamb_i
# 	# 		break
# 	# 	fi
# 	# 	lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# done
	
# 	# past_lamb="$past_lamb,$best_lamb"
	
# 	# # if [ "$id" -eq 1 ]; then
# 	# 	# elasticity_up_max_lamb=`cat ${res_path}${id}_min_lamb_w_newtask_zero.txt`
# 	# # fi

# 	best_lr_id=1
# 	best_lamb_i=19
# 	best_lamb=75.04731765
	
# 	la_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.1/"
	
# 	## Lamb Down
# 	lamb_down=1.0
# 	elasticity_up_mult=0.1
# 	alpha_lamb_i=8
# 	found_best=false
# 	while [ $found_best=false ]
# 	do
# 		((alpha_lamb_i++))
# 		custom_lr=$past_lr
# 		custom_lamb=$past_lamb
# 		printf "\n\nLA Phase\n\n"
# 		mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 		python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 		found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 		python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 		# if [ $found_best = found ]; then
# 		# 	best_alpha_lamb_i=$alpha_lamb_i
# 		# 	break
# 		# fi
# 		# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 		# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 		break
# 	done

# 	## Lamb Down
# 	lamb_down=0.1
# 	elasticity_up_mult=0.01
# 	alpha_lamb_i=9
# 	found_best=false
# 	while [ $found_best=false ]
# 	do
# 		((alpha_lamb_i++))
# 		custom_lr=$past_lr
# 		custom_lamb=$past_lamb
# 		printf "\n\nLA Phase\n\n"
# 		mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 		python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 		found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 		python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 		# if [ $found_best = found ]; then
# 		# 	best_alpha_lamb_i=$alpha_lamb_i
# 		# 	break
# 		# fi
# 		# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 		# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 		break
# 	done
		
# 	start_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.${best_alpha_lamb_i}/"
# done

id_array=(4)
for id in "${id_array[@]}"
do
	# printf "\n\nRunning search for task $id\n\n"
	# lr_id=0
	# for lr in "${lr_array[@]}"
	# do
	# 	((lr_id++))
	# 	printf "\n\nLR Iteration $lr\n\n"
	# 	custom_lamb="$past_lamb,0"
	# 	custom_lr="$past_lr,$lr"
	# 	mkdir -p  ${res_path}${id}_gold.${lr_id}/
	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
	# done
	
	# python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
	# best_lr_id=$?
	# best_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
	# past_lr="$past_lr,$best_lr"
	# python3 FABR/calc_max_lamb.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --best_lr_id $best_lr_id --best_lr $best_lr --tid $id --tid $id --custom_max_lamb $custom_max_lamb
	# start_lamb=$(<${res_path}${id}_gold_max_lamb.txt)
	# if [ "$id" -gt 1 ]; then
	# 	start_lamb=$best_lamb
	# fi

	# ## Lamb
	# lamb=$start_lamb
	# lamb_i=0
	# found_best=false
	# while [ $found_best=false ]
	# do
	# 	((lamb_i++))
	# 	custom_lr=$past_lr
	# 	custom_lamb="$past_lamb,$lamb"
	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
	# 	if [ $found_best = found ]; then
	# 		best_lamb=$lamb
	# 		best_lamb_i=$lamb_i
	# 		break
	# 	fi
	# 	lamb=`cat ${res_path}${id}_next_lamb.txt`
	# done
	
	# past_lamb="$past_lamb,$best_lamb"
	
	# # if [ "$id" -eq 1 ]; then
	# 	# elasticity_up_max_lamb=`cat ${res_path}${id}_min_lamb_w_newtask_zero.txt`
	# # fi

	best_lr_id=1
	best_lamb_i=15
	best_lamb=17.1684191
	
	la_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.1/"
	
	# ## Lamb Down
	# lamb_down=0.1
	# elasticity_up_mult=0.1
	# alpha_lamb_i=38
	# found_best=false
	# while [ $found_best=false ]
	# do
	# 	((alpha_lamb_i++))
	# 	custom_lr=$past_lr
	# 	custom_lamb=$past_lamb
	# 	printf "\n\nLA Phase\n\n"
	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
	# 	# if [ $found_best = found ]; then
	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
	# 	# 	break
	# 	# fi
	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
	# 	break
	# done

	# ## Lamb Down
	# lamb_down=0.1
	# elasticity_up_mult=0.01
	# alpha_lamb_i=39
	# found_best=false
	# while [ $found_best=false ]
	# do
	# 	((alpha_lamb_i++))
	# 	custom_lr=$past_lr
	# 	custom_lamb=$past_lamb
	# 	printf "\n\nLA Phase\n\n"
	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
	# 	# if [ $found_best = found ]; then
	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
	# 	# 	break
	# 	# fi
	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
	# 	break
	# done

	# ## Lamb Down
	# lamb_down=0.1
	# elasticity_up_mult=0.001
	# alpha_lamb_i=40
	# found_best=false
	# while [ $found_best=false ]
	# do
	# 	((alpha_lamb_i++))
	# 	custom_lr=$past_lr
	# 	custom_lamb=$past_lamb
	# 	printf "\n\nLA Phase\n\n"
	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
	# 	# if [ $found_best = found ]; then
	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
	# 	# 	break
	# 	# fi
	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
	# 	break
	# done

	## Lamb Down
	lamb_down=0.1
	elasticity_up_mult=0.01
	alpha_lamb_i=41 # Using t3.19.LA_phase.9
	found_best=false
	while [ $found_best=false ]
	do
		((alpha_lamb_i++))
		custom_lr=$past_lr
		custom_lamb=$past_lamb
		printf "\n\nLA Phase\n\n"
		mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment sent_mix --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario dil --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
		python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
		found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
		python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
		# if [ $found_best = found ]; then
		# 	best_alpha_lamb_i=$alpha_lamb_i
		# 	break
		# fi
		# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
		# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
		break
	done
		
	start_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.${best_alpha_lamb_i}/"
done

######################################################################
# bash sentmix_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg-rand0-0.05adt.sh random0 0 0 500.0 10000.0 1.0 1.0 0.8 True 0.05
# bash sentmix_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg-rand0-0.05adt.sh random0 0 0 500.0 500.0 1.0 1.0 0.8 True 0.05 # For t4 set lambda_max=500