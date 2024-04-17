#!/usr/bin/env bash

# set experiment variables
# careful: test set must match model - because of different spm dictionaries

# Must have the following variables set:
# $SRC - source language
# $TGT - target language
# $MODEL - model name
# $DATA - test set name


# $OUTPUT_DIR - output directory (arg)

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 OUTPUT_DIR"
    echo "Must have the following variables set:"
    echo "  \$SRC - source language"
    echo "  \$TGT - target language"
    echo "  \$MODEL - model name"
    echo "  \$DATA - dataset name"
    exit 1
fi

if [ -z "$DATA" ]; then
    echo "Error: DATA is not set"
    exit 1
fi

OUTPUT_DIR="$1/generation-$MODEL"

echo "Generating translations for $SRC-$TGT on $DATA"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR
output_file="$OUTPUT_DIR/$DATA-$SRC-$TGT"

# generation command
fairseq-generate Data/$DATA/bin \
     --gen-subset train --source-lang $SRC --target-lang $TGT \
     --path ./Models/$MODEL/checkpoint_best.pt \
     --skip-invalid-size-inputs-valid-test \
     --batch-size 128 --beam 5 --remove-bpe sentencepiece > $output_file

# Process generated outputs
# Hypothesis - the translated file (this will be our new source)
cat $output_file | grep -p ^H | sort -V | cut -f3- | sacremoses detokenize > $output_file.hyp

# Reference - the "reference" translation, in our case the files would be completely unrelated so we can discard that
# cat $output_file | grep -p ^T | sort -V | cut -f2- | sacremoses detokenize > $output_file.ref

# Source (this will be our new target)
cat $output_file | grep -p ^S | sort -V | cut -f2- | sacremoses detokenize > $output_file.src