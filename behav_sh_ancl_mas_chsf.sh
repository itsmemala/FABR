#!/bin/bash

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 !python  FABR//run.py --bert_model 'bert-base-uncased' --experiment annomi --approach bert_adapter_ewc_ancl --imp function --baseline ewc_ancl --backbone bert_adapter --note random0 --idrandom 0 --seed 0 --scenario dil --use_cls_wgts True --train_batch_size 32 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --learning_rate 0.00003 --custom_lamb 0,0 --custom_alpha_lamb 0,0 --ancl True --break_after_task 1 --my_save_path ~/fabr_data/BehavSH/BehavSH_ANCLMAS_t1gold/
done


