#!/usr/bin/env bash

# NOTE: Required variables to be set:
# $l1 - source language
# $l2 - target language
# $spm - path to sentencepiece model
# $src_train - path to source training data
# $tgt_train - path to target training data
# $train_file - path to target training data
# $dev_file - path to target development data
# $test_file - path to target test data

if [ -z "$l1" ]; then
    echo "l1 is not set"
    exit 1
fi

if [ -z "$l2" ]; then
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

echo "Preprocessing data for $l1-$l2"
echo "Sentencepiece model: $spm"
echo "Source training data: $src_train"
echo "Target training data: $tgt_train"
echo "Train file: $train_file"
echo "Dev file: $dev_file"
echo "Test file: $test_file"

# preprocessing

# tokenize train-mono, dev, test
cat $src_train | sacremoses -l $l1 -j 4 normalize -c tokenize -a > $train_file.tok.$l1
cat $tgt_train | sacremoses -l $l2 -j 4 normalize -c tokenize -a > $train_file.tok.$l2

cat $dev_file.$l1 | sacremoses -l $l1 -j 4 normalize -c tokenize -a > $dev_file.tok.$l1
cat $dev_file.$l2 | sacremoses -l $l2 -j 4 normalize -c tokenize -a > $dev_file.tok.$l2

cat $test_file.$l1 | sacremoses -l $l1 -j 4 normalize -c tokenize -a > $test_file.tok.$l1
cat $test_file.$l2 | sacremoses -l $l2 -j 4 normalize -c tokenize -a > $test_file.tok.$l2


# separated for clarity
python ./spm_encode.py --model="$spm" \
    --output_format=piece \
    --inputs $train_file.tok.$l1 $train_file.tok.$l2  \
    --outputs  $train_file.tok.spm.$l1 $train_file.tok.spm.$l2

python ./spm_encode.py --model="$spm" \
    --output_format=piece \
    --inputs $dev_file.tok.$l1 $dev_file.tok.$l2  \
    --outputs  $dev_file.tok.spm.$l1 $dev_file.tok.spm.$l2

python ./spm_encode.py --model="$spm" \
    --output_format=piece \
    --inputs $test_file.tok.$l1 $test_file.tok.$l2  \
    --outputs  $test_file.tok.spm.$l1 $test_file.tok.spm.$l2