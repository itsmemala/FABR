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
dataset='hwu64'
lr_array=(0.00003) # 0.0003) #(0.00003 0.0003 0.003 0.03)
decay=0.9
acc_drop_threshold=${10}
growth=0.9
base_res_path="/home/local/data/ms/fabr_data/IntentSH/IntentSH_LAMAS_NoL1_Custom_ssFixedLrP10_${pdm_frac}pdmfrac${no_frel_cut_max}_AvgPoolNoLAReg/${note}seed${seed}_${acc_drop_threshold}adt/IntentSH_LAMAS_t"
res_path="/home/local/data/ms/fabr_data/IntentSH/IntentSH_LAMAS_NoL1_Custom_Analysis1/${note}seed${seed}_${acc_drop_threshold}adt/IntentSH_LAMAS_t"
# res_path="/home/local/data/ms/fabr_data/IntentSH/IntentSH_LAMAS_NoL1_Custom_ssFixedLrP10_${pdm_frac}pdmfrac${no_frel_cut_max}_AvgPoolNoLAReg/${note}seed${seed}_${acc_drop_threshold}adt/IntentSH_LAMAS_t"

# id=0
# printf "\n\nRunning search for task 0\n\n"
# lr_id=0
# for lr in "${lr_array[@]}"
# do
# 	((lr_id++))
# 	printf "\n\nLR Iteration $lr\n\n"
# 	mkdir -p  ${res_path}${id}_gold.${lr_id}/
# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --learning_rate $lr --fisher_combine avg --break_after_task 0 --my_save_path ${res_path}${id}_gold.${lr_id}/ --only_mcl True
# done

# python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
# best_lr_id=$?
# past_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
# past_lamb=0

past_lr=0.0003,0.0003,0.0003,0.0003,0.0003 #t2/t3: 0.00003 (low FWT)
past_lamb=0,28.24295365,13.5,13.5,13.5 #t2: 13.5 (FWT not enough) #t3: 13.5 (poor FWT)/100
best_lamb=100
# start_model_path="${base_res_path}0_gold.2/"
# start_model_path="${base_res_path}1.1.LA_phase.3/"
# start_model_path="${res_path}2.6.LA_phase.41/" #start_model_path="${base_res_path}2.6.LA_phase.38/" # Note: Accidnetally overwrote a couple (.35,.36) Analysis1/ results into base_res_path
start_model_path="${res_path}3.2.LA_phase.13/"

# id_array=(1)
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
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
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

# 	# # Lamb
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
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
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
# 	best_lamb_i=1
# 	best_lamb=28.24295365

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
# 		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
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
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
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

# 	# # Lamb
# 	# lamb=0.9
# 	# lamb_i=24
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done
	
# 	# past_lamb="$past_lamb,$best_lamb"
	
# 	# if [ "$id" -eq 1 ]; then
# 		# elasticity_up_max_lamb=`cat ${res_path}${id}_min_lamb_w_newtask_zero.txt`
# 	# fi
	
# 	best_lr_id=1
# 	best_lamb_i=6
# 	best_lamb=13.5

# 	la_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.1/"
	
# 	# ## Lamb Down
# 	# lamb_down=0.000001 #0.0001
# 	# elasticity_up_mult=0.001 #0.001
# 	# alpha_lamb_i=42 #40
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	if [ $found_best = found ]; then
# 	# 		best_alpha_lamb_i=$alpha_lamb_i
# 	# 		break
# 	# 	fi
# 	# 	lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# done

# 	##############################################################
# 	# Perform analysis to get expected FWT

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=0
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=1
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=0.01
# 	# alpha_lamb_i=2
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=0.001
# 	# alpha_lamb_i=3
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=0.0001
# 	# alpha_lamb_i=4
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=5
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=6
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=0.01
# 	# alpha_lamb_i=7
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=0.001
# 	# alpha_lamb_i=8
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=0.0001
# 	# alpha_lamb_i=9
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=10
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=11
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.05
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=12
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.05
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=13
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=14
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=15
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ##########################
# 	# # Reduce lamb to 6.5 (not expecting this do well based on lamb_results.png - just a check)

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=16
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=17
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	##########################
# 	# Reduce lamb to 1.8

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=18
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=19
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ##########################
# 	# # Reduce lamb to 0.9 (this is too low)

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=20
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=21
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	##########################
# 	# Set lamb to 1.8, lamb_max=28.2

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=22
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=23
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=24
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=25
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=26
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=27
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.05
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=28
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.05
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=29
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=30
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=31
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	########################
# 	# Reduce lamb to 0.9, this time also reduce lamb_up to 0.01 to actually ensure less regularisation (lamb_max remains 28.2)
# 	# prev best is (lamb_down=0.1, lamb_up=0.1)

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=32
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.01
# 	# alpha_lamb_i=33
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	########################
# 	# Go back to best lamb (= 13.5), this time also reduce lamb_up to 0.01 to actually ensure less regularisation (lamb_max remains 28.2)
# 	# prev best is (lamb_down=0.1, lamb_up=0.1)
# 	# Accidentally overwrote first two results into base_res_path

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=34
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.01
# 	# alpha_lamb_i=35
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=34
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=0.01
# 	# alpha_lamb_i=35
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.001
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=36
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.001
# 	# elasticity_up_mult=0.01
# 	# alpha_lamb_i=37
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ################################
# 	## Run with higher lr

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=0.01
# 	# alpha_lamb_i=38
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=39
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=40
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=41
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# Increase lamb from 13.5 -> 28.2
# 	## Lamb Down
# 	lamb_down=0.01
# 	elasticity_up_mult=1.0
# 	alpha_lamb_i=42
# 	found_best=false
# 	while [ $found_best=false ]
# 	do
# 		((alpha_lamb_i++))
# 		custom_lr=$past_lr
# 		custom_lamb=$past_lamb
# 		printf "\n\nLA Phase\n\n"
# 		mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
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

# 	# start_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.${best_alpha_lamb_i}/"
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
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
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

# 	# best_lr_id=1

# 	# ## Lamb
# 	# lamb=28.24295365
# 	# lamb_i=4
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done

# 	# ## Lamb
# 	# lamb=50.0
# 	# lamb_i=5
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done

# 	# ## Lamb
# 	# lamb=100.0
# 	# lamb_i=6
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done

# 	# ## Lamb
# 	# lamb=175.0
# 	# lamb_i=8
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done
	
# 	######################
# 	# Start from Analysis1/ * t2.6.LA_phase.36/ path

# 	# ## Lamb
# 	# lamb=300.0
# 	# lamb_i=0
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done

# 	# ## Lamb
# 	# lamb=100.0
# 	# lamb_i=1
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done

# 	# ## Lamb
# 	# lamb=75.0
# 	# lamb_i=2
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done

# 	# ## Lamb
# 	# lamb=50.0
# 	# lamb_i=3
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb="$past_lamb,$lamb"
# 	# 	printf "\n\nLamb Iteration $custom_lamb \n\n"
# 	# 	mkdir -p  ${res_path}${id}.${lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
# 	# 	python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
# 	# 	python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_lamb=$lamb
# 	# 	# 	best_lamb_i=$lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb=`cat ${res_path}${id}_next_lamb.txt`
# 	# 	break
# 	# done

# 	# past_lamb="$past_lamb,$best_lamb"
	
# 	# if [ "$id" -eq 1 ]; then
# 		# elasticity_up_max_lamb=`cat ${res_path}${id}_min_lamb_w_newtask_zero.txt`
# 	# fi
	
# 	# best_lr_id=1
# 	# best_lamb_i=1
# 	# best_lamb=13.5

# 	# best_lr_id=1
# 	# best_lamb_i=7
# 	# best_lamb=100

# 	best_lr_id=1
# 	best_lamb_i=2
# 	best_lamb=100

# 	la_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.1/"
	
# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=0
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=1
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=2
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=3
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=0.5
# 	# alpha_lamb_i=4
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=0.5
# 	# alpha_lamb_i=5
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.5
# 	# alpha_lamb_i=6
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=0.5
# 	# alpha_lamb_i=7
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=8
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.5
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=9
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.1
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=10
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# ## Lamb Down
# 	# lamb_down=0.01
# 	# elasticity_up_mult=0.1
# 	# alpha_lamb_i=11
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	######################
# 	# Set lamb from 100 -> 13.5 and lamb_max from 246 -> 28.2

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=12
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# Since we still have high new task perf, set lamb_max to 246

# 	# ## Lamb Down
# 	# lamb_down=1.0
# 	# elasticity_up_mult=1.0
# 	# alpha_lamb_i=13
# 	# found_best=false
# 	# while [ $found_best=false ]
# 	# do
# 	# 	((alpha_lamb_i++))
# 	# 	custom_lr=$past_lr
# 	# 	custom_lamb=$past_lamb
# 	# 	printf "\n\nLA Phase\n\n"
# 	# 	mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
# 	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 35 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
# 	# 	python3 FABR/calc_next_lamb_down_lamb_up.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --elasticity_up_mult $elasticity_up_mult --growth $growth --tid $id
# 	# 	found_best=`cat ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}_foundbestlambdown.txt`
# 	# 	python3 FABR/plot_lamb_down_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --lamb_down $lamb_down --tid $id
# 	# 	# if [ $found_best = found ]; then
# 	# 	# 	best_alpha_lamb_i=$alpha_lamb_i
# 	# 	# 	break
# 	# 	# fi
# 	# 	# lamb_down=`cat ${res_path}${id}_next_lamb_down.txt`
# 	# 	# elasticity_up_mult=`cat ${res_path}${id}_next_lamb_up.txt`
# 	# 	break
# 	# done

# 	# start_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.${best_alpha_lamb_i}/"
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
	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
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
	# 	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path --only_mcl True
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

	best_lr_id=1 #Skipped running this
	best_lamb_i=1  #Skipped running this
	best_lamb=13.5
	
	la_model_path="${res_path}${id}.${best_lamb_i}.LA_phase.1/"
	
	## Lamb Down
	lamb_down=1.0
	elasticity_up_mult=1.0
	alpha_lamb_i=0
	found_best=false
	while [ $found_best=false ]
	do
		((alpha_lamb_i++))
		custom_lr=$past_lr
		custom_lamb=$past_lamb
		printf "\n\nLA Phase\n\n"
		mkdir -p ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/
		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_freeze --imp function --baseline ewc_freeze --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 20 --custom_lr $custom_lr --custom_lamb $custom_lamb --fisher_combine avg --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.LA_phase.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path --elasticity_down_max_lamb $elasticity_up_max_lamb --elasticity_down_mult $elasticity_up_mult --elasticity_up $lamb_down --frel_cut_type pdm --pdm_frac $pdm_frac --no_frel_cut_max $no_frel_cut_max --la_model_path $la_model_path --no_reg_in_LA True
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

# CUDA_VISIBLE_DEVICES=1 bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random0 0 0 0.32346185 1641.28483697 1.0 1.0 0.9 True 0.3
# CUDA_VISIBLE_DEVICES=1 bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random3 3 0 6.85163861 77.30662811 1.0 1.0 0.9 True 0.3
# CUDA_VISIBLE_DEVICES=0 bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random6 6 0 28.24295365 246.34804902 1.0 1.0 0.9 True 0.3

# CUDA_VISIBLE_DEVICES=0 bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random0 0 0 0.32346185 1641.28483697 1.0 1.0 0.9 True 0.2
# CUDA_VISIBLE_DEVICES=1 bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random3 3 0 6.85163861 77.30662811 1.0 1.0 0.9 True 0.2
# CUDA_VISIBLE_DEVICES=1 bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random6 6 0 28.24295365 246.34804902 1.0 1.0 0.9 True 0.2

# bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random0 0 0 0.04854989 1641.28483697 1.0 1.0 0.9 True 0.1
# bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random3 3 0 4.49536009 77.30662811 1.0 1.0 0.9 True 0.1
# bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random6 6 0 28.24295365 246.34804902 1.0 1.0 0.9 True 0.1

###############################################################
# bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random3 3 0 4.49536009 77.30662811 1.0 1.0 0.8 True 0.1
# bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg-rand6.sh random6 6 0 28.24295365 246.34804902 1.0 1.0 0.8 True 0.1 # for t2: set lamb_max=28.2

###############################################################
# bash intent_sh_la_mas_chsf_nol1_custom_searchlambup_searchlambdown_pdmfrac-avgpool-nolareg.sh random3 3 0 4.49536009 77.30662811 1.0 1.0 0.7 True 0.1