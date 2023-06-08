import sys,os,argparse,time
import numpy as np
import torch
import multiprocessing
import utils

def asc_config(parser):
    # EWC args
    # Source: https://github.com/ZixuanKe/PyContinual/blob/54dd15de566b110c9bc8d8316205de63a4805190/src/config.py
    parser.add_argument('--lamb',default=5000,type=float,required=False,help='(default=%(default)f)')
    # DER++ args
    # Source: https://github.com/ZixuanKe/PyContinual/blob/54dd15de566b110c9bc8d8316205de63a4805190/src/config.py
    # Defaults: https://github.com/ZixuanKe/PyContinual/blob/54dd15de566b110c9bc8d8316205de63a4805190/src/load_base_args.py
    parser.add_argument('--backbone',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--baseline',default='',type=str,required=True,help='(default=%(default)s)')
    parser.add_argument('--buffer_percent', default=0.02, type=float, required=False,
                        help='The size of the memory buffer.')
    parser.add_argument('--buffer_size', default=128, type=int, required=False,
                        help='The size of the memory buffer.')
    parser.add_argument('--alpha', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default=0.5, type=float, help='(default=%(default)f)')
    # TaskDrop args
    # Source: https://github.com/zylMozart/TaskDrop/blob/main/config.py
    parser.add_argument('--multi_mask',type=int,default=1,help='for random mask only')
    # CTR args
    # Source: https://github.com/ZixuanKe/PyContinual/blob/54dd15de566b110c9bc8d8316205de63a4805190/src/config.py
    parser.add_argument('--build_adapter', action='store_true')
    parser.add_argument('--build_adapter_ucl', action='store_true')
    parser.add_argument('--build_adapter_owm', action='store_true')
    parser.add_argument('--build_adapter_mask', action='store_true')
    parser.add_argument('--build_adapter_capsule_mask', action='store_true')
    parser.add_argument('--build_adapter_capsule', action='store_true')
    parser.add_argument('--apply_bert_output', action='store_true')
    parser.add_argument('--apply_bert_attention_output', action='store_true')
    parser.add_argument('--apply_one_layer_shared', action='store_true')
    parser.add_argument('--use_imp', action='store_true')
    parser.add_argument('--use_gelu', action='store_true')
    parser.add_argument('--transfer_route',action='store_true')
    parser.add_argument('--share_conv', action='store_true')
    parser.add_argument('--larger_as_share',action='store_true')
    parser.add_argument('--bert_mask_adapter_size',default=2000,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--bert_adapter_size',default=2000,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--semantic_cap_size',default=3,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--num_semantic_cap',default=3,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--larger_as_list', action='store_true')
    parser.add_argument('--no_tsv_mask', action='store_true')
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--temp', type=float, default=1,
                        help='temperature for loss function')
    parser.add_argument('--base_temp', type=float, default=1,
                        help='temperature for loss function')
    parser.add_argument('--scenario',default='',type=str,required=True,help='(default=%(default)s)')
    # FABR dev args
    parser.add_argument('--use_rbs',default=False,type=bool,required=False,help='(default=%(default)s)')
    parser.add_argument('--regularize_t0',default=False,type=bool,required=False,help='(default=%(default)s)')
    parser.add_argument('--imp',default='loss',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--l1_lamb',default=0.000001,type=float,required=False,help='(default=%(default)s)')
    parser.add_argument('--use_l1',default=False,type=bool,required=False,help='(default=%(default)s)')
    parser.add_argument('--freeze_cutoff',default=0.99,type=float,required=False,help='(default=%(default)s)')
    parser.add_argument('--mask_scaling',default=0.1,type=float,required=False,help='(default=%(default)s)')
    parser.add_argument('--elasticity_down',default=10,type=float,required=False,help='(default=%(default)s)')
    parser.add_argument('--elasticity_up',default=0.1,type=float,required=False,help='(default=%(default)s)')
    parser.add_argument('--my_save_path',default='',type=str,required=True,help='(default=%(default)s)')
    parser.add_argument('--fisher_combine',default='avg',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--convert_to_binary',default=None,type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--subset_data',default=None,type=int,required=False,help='(default=%(default)s)')
    parser.add_argument('--fa_method',default='ig',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--transfer_acc',default=False,type=bool,required=False,help='(default=%(default)s)')
    parser.add_argument('--withcontext',default=False,type=bool,required=False,help='(default=%(default)s)')
    parser.add_argument('--use_cls_wgts',default=False,type=bool,required=False,help='(default=%(default)s)')
    parser.add_argument('--mlp_depth',default=1,type=int,required=False,help='(default=%(default)s)')
    parser.add_argument('--save_metadata',default=None,type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--valid_loss_es',default=0.0,type=float,required=False,help='(default=%(default)s)')
    parser.add_argument('--lr_patience',default=5,type=int,required=False,help='(default=%(default)s)')
    parser.add_argument('--lfa_lambda',default=1.0,type=float,required=False,help='(default=%(default)s)')
    parser.add_argument('--lfa',default=None,type=str,required=False,help='(default=%(default)s)')
    # KAN args
    # Source: 
    parser.add_argument('--experiment',default='',type=str,required=True,help='(default=%(default)s)')
    parser.add_argument('--approach',default='',type=str,required=True,help='(default=%(default)s)')
    parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--note',type=str,default='',help='(default=%(default)d)')
    parser.add_argument('--ntasks',default=10,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--idrandom',default=10,type=int,required=False,help='(default=%(default)d)')
    parser.add_argument('--output_dir',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument("--num_task", default=19, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--mask_whole_word', action='store_true',
                        help="Whether masking a whole word.")
    parser.add_argument('--max_pred', type=int, default=128,
                        help="Max tokens of prediction.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--beam_size', type=int, default=5,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--not_predict_token', type=str, default=None,
                        help="Do not predict the tokens during decoding.")
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warm_train", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    return parser

def train_config(parser):
    ## Other parameters

    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--bert_hidden_size", default=768, type=str)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_term_length",
                        default=5,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_train_epochs",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")

    # attention
    parser.add_argument('--deep_att_lexicon_input_on', action='store_false')
    parser.add_argument('--deep_att_hidden_size', type=int, default=64)
    parser.add_argument('--deep_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--deep_att_activation', type=str, default='relu')
    parser.add_argument('--deep_att_norm_on', action='store_false')
    parser.add_argument('--deep_att_proj_on', action='store_true')
    parser.add_argument('--deep_att_residual_on', action='store_true')
    parser.add_argument('--deep_att_share', action='store_false')
    parser.add_argument('--deep_att_opt', type=int, default=0)

    # self attn
    parser.add_argument('--self_attention_on', action='store_false')
    parser.add_argument('--self_att_hidden_size', type=int, default=64)
    parser.add_argument('--self_att_sim_func', type=str, default='dotproductproject')
    parser.add_argument('--self_att_activation', type=str, default='relu')
    parser.add_argument('--self_att_norm_on', action='store_true')
    parser.add_argument('--self_att_proj_on', action='store_true')
    parser.add_argument('--self_att_residual_on', action='store_true')
    parser.add_argument('--self_att_dropout', type=float, default=0.1)
    parser.add_argument('--self_att_drop_diagonal', action='store_false')
    parser.add_argument('--self_att_share', action='store_false')
    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.4)
    # query summary
    parser.add_argument('--query_sum_att_type', type=str, default='linear',
                        help='linear/mlp')
    parser.add_argument('--query_sum_norm_on', action='store_true')
    parser.add_argument('--decoder_ptr_update_on', action='store_true')
    parser.add_argument('--decoder_num_turn', type=int, default=5)
    parser.add_argument('--decoder_mem_type', type=int, default=3)
    parser.add_argument('--decoder_mem_drop_p', type=float, default=0.2)
    parser.add_argument('--decoder_opt', type=int, default=0)
    parser.add_argument('--decoder_att_type', type=str, default='bilinear',
                        help='bilinear/simple/default')
    parser.add_argument('--decoder_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/default')
    parser.add_argument('--decoder_weight_norm_on', action='store_true')
    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = asc_config(parser)
    parser = train_config(parser)

    args = parser.parse_args()
    return args
