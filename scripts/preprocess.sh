#!/usr/bin/env bash

# NOTE: Required variables to be set:
# $SRC - source language
# $TGT - target language
# $spm - path to sentencepiece model
# $src_train - path to source training data
# $tgt_train - path to target training data
# $train_file - path to target training data
# $dev_file - path to target development data
# $test_file - path to target test data

if [ -z "$SRC" ]; then
    echo "l1 is not set"
    exit 1
fi

if [ -z "$TGT" ]; then
    echo "l2 is not set"
    exit 1
fi

if [ -z "$spm" ]; then
    echo "spm is not set"
    exit 1
fi

if [ -z "$src_train" ]; then
    echo "src_train is not set"
    exit 1
fi

if [ -z "$tgt_train" ]; then
    echo "tgt_train is not set"
    exit 1
fi

if [ -z "$train_file" ]; then
    echo "train_file is not set"
    exit 1
fi

if [ -z "$dev_file" ]; then
    echo "dev_file is not set"
    exit 1
fi

if [ -z "$test_file" ]; then
    echo "test_file is not set"
    exit 1
fi

echo "Preprocessing data for $SRC-$TGT"
echo "Sentencepiece model: $spm"
echo "Source training data: $src_train"
echo "Target training data: $tgt_train"
echo "Train file: $train_file"
echo "Dev file: $dev_file"
echo "Test file: $test_file"

# preprocessing

# tokenize train-mono, dev, test
cat $src_train | sacremoses -l $SRC -j 4 normalize -c tokenize -a > $train_file.tok.$SRC
cat $tgt_train | sacremoses -l $TGT -j 4 normalize -c tokenize -a > $train_file.tok.$TGT

cat $dev_file.$SRC | sacremoses -l $SRC -j 4 normalize -c tokenize -a > $dev_file.tok.$SRC
cat $dev_file.$TGT | sacremoses -l $TGT -j 4 normalize -c tokenize -a > $dev_file.tok.$TGT

cat $test_file.$SRC | sacremoses -l $SRC -j 4 normalize -c tokenize -a > $test_file.tok.$SRC
cat $test_file.$TGT | sacremoses -l $TGT -j 4 normalize -c tokenize -a > $test_file.tok.$TGT


# separated for clarity
python ./spm_encode.py --model="$spm" \
    --output_format=piece \
    --inputs $train_file.tok.$SRC $train_file.tok.$TGT  \
    --outputs  $train_file.tok.spm.$SRC $train_file.tok.spm.$TGT

python ./spm_encode.py --model="$spm" \
    --output_format=piece \
    --inputs $dev_file.tok.$SRC $dev_file.tok.$TGT  \
    --outputs  $dev_file.tok.spm.$SRC $dev_file.tok.spm.$TGT

python ./spm_encode.py --model="$spm" \
    --output_format=piece \
    --inputs $test_file.tok.$SRC $test_file.tok.$TGT  \
    --outputs  $test_file.tok.spm.$SRC $test_file.tok.spm.$TGT