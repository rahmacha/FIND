#!/usr/bin/env python

import sweep
from sweep import hyperparam

def get_grid(args):
    return [
        #hyperparam('--fp16'),
        #hyperparam('--ddp-backend', 'no_c10d'),
        hyperparam('--max-epoch', 200),
        #hyperparam('--max-update', 300000),

        # equivalent to training on 16x GPUs
        hyperparam('--update-freq', 16),

        hyperparam('--arch', 'transformer_iwslt_de_en'),

        hyperparam('--optimizer', 'adam'),
        hyperparam('--adam-betas', '(0.9, 0.98)'),
        hyperparam('--lr-scheduler', 'inverse_sqrt'),
        hyperparam('--lr', [0.005], save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--min-lr', 1e-9),
        hyperparam('--clip-norm-MT', 0.0),
        hyperparam("--weight-decay", 0.0001),

        hyperparam('--criterion', 'label_smoothed_cross_entropy'),
        hyperparam('--label-smoothing-MT', [0.0], save_dir_key=lambda val: f'smooth{val}'),

        hyperparam('--max-tokens', [4096], save_dir_key=lambda val: f'maxtok{val}'),
        hyperparam('--seed', [1,2,3,4], save_dir_key=lambda val: f'seed{val}'),

        hyperparam('--log-format', 'json'),

        #hyperparam('--eval-bleu'),
        #hyperparam('--eval-bleu-detok', 'moses'),

        hyperparam('--mdl-block-size', 4),
        hyperparam('--mdl-batches-per-epoch', 3000),
        hyperparam('--mdl-train-examples', 4),

        hyperparam('--data-bias', ['4_linear', '4_hierar'], save_dir_key=lambda val: f'bias{val}'),
        hyperparam('--data-pretraining', ['pretrain_large_linear', 'pretrain_large_hierar'], save_dir_key=lambda val: f'set{val}'),
        #hyperparam('--data-bias', ['10_mem', '10_count'], save_dir_key=lambda val: f'bias{val}'),
        #hyperparam('--data-pretraining', ['pretrain_large_count', 'pretrain_large_mem', ''], save_dir_key=lambda val: f'set{val}'),

        hyperparam('--dropout-MT', 0.3),
        hyperparam('--dropout-bias', 0.3),
        hyperparam("--attention-dropout", 0.1),
        hyperparam('--warmup-init-lr-MT', 1e-7),
        hyperparam('--warmup-updates-MT', 7000),
        hyperparam('--warmup-init-lr-bias', [1e-7], save_dir_key=lambda val: f'warminitlr{val}'),
        hyperparam('--warmup-updates-bias', [7000], save_dir_key=lambda val: f'warmup{val}'),
    ]

def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
