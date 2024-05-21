#!/usr/bin/env bash

# ##############################################################################
# The following script is used to run the experiments for the first part of the project
# It performs the following experiments:
# - Test base MODEL performance on the test set
# - Test base MODEL performance on the it-parallel dataset
# - Finetune the base MODEL on the it-parallel dataset and evaluate on both test sets
# ###############################################################################
# Set the default variables for the experiment:
SRC=en
TGT=de
MODEL=big-$SRC-$TGT
TEST_DATA_NEWS=train-euro-news-big.$SRC-$TGT
TEST_DATA_PARA=it-parallel
OUTPUT_DIR=tests

################################################################################

TEST=$TEST_DATA_NEWS

# generate
source scripts/generate.sh
# evaluate
source scripts/evaluate.sh

################################################################################
# IT-PARALLEL
################################################################################

TEST=$TEST_DATA_PARA

# generate
source scripts/generate.sh
# evaluate
source scripts/evaluate.sh


################################################################################
# FINETUNE ON IT-PARALLEL
################################################################################

data=it-parallel
experiment=ft-$model-$data-$SRC-$TGT
fairseq-train \
    "Data/$data/bin" \
    --finetune-from-model Models/$MODEL/checkpoint_best.pt \
    --arch transformer_wmt_en_de \
    --task translation \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.1 \
    --lr 0.0006 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 2500 \
    --warmup-init-lr 1e-07 \
    --stop-min-lr 1e-09 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 8192 \
    --max-update 2 \
    --update-freq 8 \
    --patience 10 \
    --scoring sacrebleu \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --save-interval-updates 2000 \
    --validate-interval-updates 2000 \
    --no-epoch-checkpoints \
    --keep-best-checkpoints 1 \
    --encoder-learned-pos \
    --save-dir Models/$experiment \
    --bpe sentencepiece

# NOTE: Switch to the best model
MODEL=$experiment

# 1) Evaluate on news
TEST=$TEST_DATA_NEWS

# generate
source scripts/generate.sh
# evaluate
source scripts/evaluate.sh

# 2) Evaluate on it-parallel
TEST=$TEST_DATA_PARA

# generate
source scripts/generate.sh
# evaluate
source scripts/evaluate.sh
