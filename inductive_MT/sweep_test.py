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

        hyperparam('--arch', 'lstm'),
        hyperparam('--decoder-attention', 1),

        hyperparam('--encoder-embed-dim', 256),
        hyperparam('--encoder-hidden-size', 512),
        hyperparam('--encoder-layers', 2),

        hyperparam('--decoder-embed-dim', 256),
        hyperparam('--decoder-hidden-size', 512),
        hyperparam('--decoder-layers', 2),

        hyperparam('--optimizer', 'adam'),
        hyperparam('--adam-betas', '(0.9, 0.98)'),
        hyperparam('--lr-scheduler', 'inverse_sqrt'),
        hyperparam('--lr', 1e-3),
        hyperparam('--min-lr', 1e-9),
        hyperparam('--clip-norm-MT', 0.1),

        hyperparam('--criterion', 'label_smoothed_cross_entropy'),
        hyperparam('--label-smoothing', 0.0),

        hyperparam('--max-tokens', 4096),
        hyperparam('--seed', [1,2,3,4], save_dir_key=lambda val: f'seed{val}'),

        hyperparam('--log-format', 'json'),

        #hyperparam('--eval-bleu'),
        #hyperparam('--eval-bleu-detok', 'moses'),

        hyperparam('--mdl-block-size', 1),
        hyperparam('--mdl-batches-per-epoch', 3000),
        hyperparam('--mdl-train-examples', 1),

        hyperparam('--data-bias', ['10_mem', '10_count'], save_dir_key=lambda val: f'bias{val}'),
        hyperparam('--data-pretraining', ['pretrain_large_count'], save_dir_key=lambda val: f'set{val}'),

        hyperparam('--dropout-MT', 0.3),
        hyperparam('--dropout-bias', 0.5),
        hyperparam('--warmup-init-lr-MT', 1e-7),
        hyperparam('--warmup-updates-MT', 7000),
        hyperparam('--warmup-init-lr-bias', 1e-5),
        hyperparam('--warmup-updates-bias', 1000),

    ]

def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
