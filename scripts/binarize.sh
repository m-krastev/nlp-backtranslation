#!/usr/bin/env bash
# binarise data for training
# Usage: bash scripts/binarize.sh <--only-source>

if [ "$#" -gt 1 ]; then
    only_source="--only-source"
fi

fairseq-preprocess \
    --source-lang $l1 --target-lang $l2 \
    --srcdict ./Data/it-mono/dict.$l1.txt \
    --tgtdict ./Data/it-mono/dict.$l2.txt \
    --trainpref $train_file.tok.spm \
        --validpref $dev_file.tok.spm \
        --testpref $test_file.tok.spm \
    --destdir $databin \
        --thresholdtgt 0 --thresholdsrc 0 --workers 20 $only_source

# NOTE: if monolingual, --only-source
# repeat in opposite direction if required, binary files are directional