# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adopted from fairseq https://github.com/pytorch/fairseq/

import sys
import os
import collections
import random
import pathlib
import json

import numpy as np
import torch

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils, distributed_utils
from fairseq_cli.train import main as task_main
from fairseq_cli.train import distributed_main as distributed_task_main

sys.path.insert(1, '/private/home/rchaabouni/FIND')
from generate import cli_main as generate_main
from mdl import main as inductive_main

def get_params(params):
    parser = options.get_training_parser()
    parser.add_argument("--mdl-block-size", type=int, default=1, 
        help="Size of the transmitted block. Used when calculating description length")
    parser.add_argument("--mdl-batches-per-epoch", type=int, default=30, help="Number of updates in per training")
    parser.add_argument("--mdl-batch-size", type=int, default=None, help="If set, specifies the number of examples sampled (with replacement) "
                "for each update of the learner. If not specified, all examples available at the step are used.")
    parser.add_argument("--mdl-train-examples", type=int, default=1, 
            help="First `mdl-train-examples`  lines in the training dataset are considered as initial training data (see README).")
    parser.add_argument("--data-pretraining", type=str, default='', 
            help="path for the pretraining task. if `` no pretraining")
    parser.add_argument("--data-bias", type=str, default='count', required=True, 
            help="path for the  bias data")
    parser.add_argument("--dropout-MT", type=float, default=0.3, 
        help="dropout probability when training on MT")
    parser.add_argument("--dropout-bias", type=float, default=0.5, 
        help="dropout probability when training on simple task and investigating biases")
    parser.add_argument("--warmup-init-lr-MT", type=float, default=1e-7)
    parser.add_argument("--warmup-updates-MT", type=float, default=7000)
    parser.add_argument("--warmup-init-lr-bias", type=float, default=1e-5)
    parser.add_argument("--warmup-updates-bias", type=float, default=1000)
    parser.add_argument("--clip-norm-MT", type=float, default=0.0)
    parser.add_argument("--label-smoothing-MT", type=float, default=0.0)

    args = options.parse_args_and_arch(parser, input_args=params)
    return args

def parallel_training(args, distributed_world_size, update_freq):
    args.distributed_world_size = distributed_world_size
    args.fp16 = True
    args.ddp_backend = 'no_c10d'
    args.update_freq = update_freq

def nonparallel_training(args):
    args.distributed_world_size = 1
    args.fp16 = False
    args.ddp_backend = 'c10d'
    args.update_freq = [1]

def measure_inductive(args):
    assert torch.cuda.is_available()
    assert args.mdl_train_examples
    # override multi-gpu logic
    inductive_main(args)

def task_training(args):
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_task_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_task_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_task_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        task_main(args)

def new_data(new_dir):
    new_path = pathlib.Path(__file__).parent.absolute() / 'task_jointdic' / new_dir / 'data-bin'
    assert new_path.exists()
    return str(new_path)

def add_savedir(save_dir, new_dir):
    tmp_dir = pathlib.Path(save_dir) / new_dir
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return str(tmp_dir)

def restore_checkpoint(args, checkpoint_path):
    args.restore_file = checkpoint_path
    args.reset_lr_scheduler, args.reset_optimizer, args.reset_dataloader, args.reset_meters = True, True, True, True

def compute_bias(args, save_dir, new_save_dir, checkpoint_path=None):
    # Update params
    if checkpoint_path:
        restore_checkpoint(args, checkpoint_path)
    args.data = new_data(args.data_bias)
    args.save_dir = add_savedir(save_dir, new_save_dir)
    if not args.sentence_avg:
        args.sentence_avg = True
    ## For bias and simple tasks
    args.disable_validation = True
    args.dropout = args.dropout_bias
    args.clip_norm = 0.0 # TODO
    args.label_smoothing = 0.0 # TODO
    args.warmup_init_lr = args.warmup_init_lr_bias
    args.warmup_updates = args.warmup_updates_bias
    args.source_lang, args.target_lang = 'src', 'dst' # TODO
    # Compute MDL
    measure_inductive(args)
    # FPA
    generate_params = [args.data, "--path=" + str(pathlib.Path(args.save_dir) / '0.pt'), '--beam=1', 
                '--batch-size=128', '--gen-subset=test']
    generate_main(generate_params)

    initial_path = str(pathlib.Path(args.save_dir) / 'initial.pt')
    return  initial_path

def simple_task_training(args, save_dir, new_save_dir, checkpoint_path):
    ## restore file and restart training parameters
    restore_checkpoint(args, checkpoint_path)
    ## Update params
    args.dropout = args.dropout_bias
    args.data = new_data(args.data_pretraining)
    args.save_dir = add_savedir(save_dir, new_save_dir)

    args.no_epoch_checkpoints = True
    args.max_epoch = 600 # TODO

    task_training(args)
    pretrained_path = str(pathlib.Path(args.save_dir) / 'checkpoint_last.pt')

    ## Verify it learned the train set 
    generate_params = [args.data, "--path=" + pretrained_path, '--beam=1', 
                '--batch-size=128', '--gen-subset=train']
    generate_main(generate_params)

    return pretrained_path

def main(args):
    args = get_params(sys.argv[1:])

    save_dir = args.save_dir

    data = args.data
    max_epoch = args.max_epoch
    sentence_avg = args.sentence_avg
    distributed_world_size = args.distributed_world_size
    update_freq = args.update_freq
    # Compute biases
    nonparallel_training(args)
    print('**** innate biases ****')
    initial_path = compute_bias(args, save_dir, 'bias_1')
    # Pretrain on simple task and recompute_biases (optional)
    if args.data_pretraining != '':
        print('**** pretraining ****')
        pretrained_path = simple_task_training(args, save_dir, 'simple_task', initial_path)
        ## compute biases after pretraining
        print('**** pretrained biases ****')
        _ = compute_bias(args, save_dir, 'bias_2', checkpoint_path=pretrained_path)

    # Train on MT
    print('**** translation ****')
    parallel_training(args, distributed_world_size, update_freq)
    ## restore file and restart training parameters
    if args.data_pretraining == '':
        restore_checkpoint(args, initial_path)
    else:
        restore_checkpoint(args, pretrained_path)
    ## Update params
    args.data = new_data(data)
    args.save_dir = add_savedir(save_dir, 'MT_task')
    args.max_epoch = max_epoch
    args.sentence_avg = sentence_avg
    args.save_interval, args.no_epoch_checkpoints = 100, False # TODO
    args.source_lang, args.target_lang = 'de', 'en' # TODO
    args.dropout = args.dropout_MT
    args.clip_norm = args.clip_norm_MT
    args.label_smoothing = args.label_smoothing_MT
    args.warmup_init_lr = args.warmup_init_lr_MT
    args.warmup_updates = args.warmup_updates_MT
    args.disable_validation, args.eval_blue, args.eval_bleu_detok = False, True, 'moses'

    task_training(args)
    checkpoint_path = str(pathlib.Path(args.save_dir) / 'checkpoint_best.pt')
    # Compute biases
    print('**** final biases ****')
    nonparallel_training(args)
    if not args.distributed_rank: # a bug in fairseq
        args.distributed_rank = 0
    _ = compute_bias(args, save_dir, 'bias_3', checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    main(sys.argv[1:])