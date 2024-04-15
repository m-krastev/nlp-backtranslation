#!/usr/bin/env bash

# TEST=it-mono
# SRC=de
# TGT=en
# exp=mid-de-en
# MODEL=mid

# Runs the evaluation script for the given model and test set
cat $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT.hyp | sacrebleu $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT.ref -m bleu > $OUTPUT_DIR/$MODEL.test-$TEST-$SRC-$TGT.sacrebleu