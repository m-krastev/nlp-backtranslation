#!/usr/bin/env bash

# set experiment variables
# careful: test set must match model - because of different spm dictionaries

# Must have the following variables set:
# $SRC - source language
# $TGT - target language
# $MODEL - model name
# $TEST - test set name
# $OUTPUT_DIR - output directory

echo "Generating translations for $SRC-$TGT on $TEST"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR

# generation command
fairseq-generate Data/$TEST/bin \
     --gen-subset test --source-lang $SRC --target-lang $TGT \
     --path ./Models/$MODEL/checkpoint_best.pt \
     --skip-invalid-size-inputs-valid-test \
     --batch-size 128 --beam 5 --remove-bpe sentencepiece > $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT

# Process generated outputs
cat $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT | grep -p ^H | sort -V | cut -f3- | sacremoses detokenize > $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT.hyp
cat $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT | grep -p ^T | sort -V | cut -f2- | sacremoses detokenize > $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT.ref
cat $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT | grep -p ^S | sort -V | cut -f2- | sacremoses detokenize > $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT.src