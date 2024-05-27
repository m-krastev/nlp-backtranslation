#!/usr/bin/env bash

# ##############################################################################
# The following script is used to run the experiments for the first part of the project
# It performs the following experiments:
# - Perform backtranslation on the it-mono dataset and evaluate on it-parallel and news corpus
# ##############################################################################

# Test base MODEL performance
SRC=en
TGT=de

TEST_DATA_NEWS=train-euro-news-big.$SRC-$TGT
TEST_DATA_PARA=it-parallel

OUTPUT_DIR=Data/backtranslation
mkdir $OUTPUT_DIR

data=it-mono
MODEL=big-$TGT-$SRC


# Copy the dev and test (binary) files

mkdir -p $OUTPUT_DIR/bin-$TGT-$SRC/
cp Data/$data/bin-$TGT-$SRC/valid* $OUTPUT_DIR/bin-$TGT-$SRC/
cp Data/$data/bin-$TGT-$SRC/test* $OUTPUT_DIR/bin-$TGT-$SRC/

# $dev_file=$OUTPUT_DIR/valid.$SRC
# $test_file=$OUTPUT_DIR/test.$SRC

# GENERATE BACKTRANSLATIONS
fairseq-generate Data/$data/bin-$TGT-$SRC \
     --gen-subset train --source-lang $TGT --target-lang $SRC \
     --path Models/$MODEL/checkpoint_best.pt \
     --skip-invalid-size-inputs-valid-test \
     --batch-size 128 --beam 5 --remove-bpe sentencepiece > $OUTPUT_DIR/$MODEL.train-$data-$TGT-$SRC

# Process generated outputs
# Hypothesis becomes new source
cat $OUTPUT_DIR/$MODEL.train-$data-$TGT-$SRC | grep ^H | sort -V | cut -f3- | sacremoses detokenize > $OUTPUT_DIR/$MODEL.train.$SRC
# Source becomes new target
cat $OUTPUT_DIR/$MODEL.train-$data-$TGT-$SRC | grep ^S | sort -V | cut -f2- | sacremoses detokenize > $OUTPUT_DIR/$MODEL.train.$TGT

train_file=$OUTPUT_DIR/$MODEL.train

# Tokenize and binarize the backtranslated data
cat $OUTPUT_DIR/$MODEL.train.$SRC | sacremoses -l $SRC -j 4 normalize -c tokenize -a > $train_file.tok.$SRC
cat $OUTPUT_DIR/$MODEL.train.$TGT | sacremoses -l $TGT -j 4 normalize -c tokenize -a > $train_file.tok.$TGT


python ./spm_encode.py --model="$spm" \
    --output_format=piece \
    --inputs $train_file.tok.$SRC $train_file.tok.$TGT  \
    --outputs  $train_file.tok.spm.$SRC $train_file.tok.spm.$TGT

fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --srcdict ./Data/it-mono/dict.$SRC.txt \
    --tgtdict ./Data/it-mono/dict.$TGT.txt \
    --trainpref $train_file.tok.spm \
        # --validpref $dev_file.tok.spm \
        # --testpref $test_file.tok.spm \
    --destdir "$(dirname $train_file)/bin" \
        --thresholdtgt 0 --thresholdsrc 0 --workers 20 --only-source

# Train the model on the backtranslated data
$MODEL=big-$SRC-$TGT
$experiment=ft-bt-$MODEL-$data
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
    --max-epoch 20 \
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

MODEL=$experiment
TEST=$TEST_DATA_NEWS

# generate
source scripts/generate.sh
# evaluate
source scripts/evaluate.sh

TEST=$TEST_DATA_PARA

# generate
source scripts/generate.sh
# evaluate
source scripts/evaluate.sh