#!/usr/bin/env bash
# binarise data for training
# Usage: bash scripts/binarize.sh <--only-source>

if [ "$#" -gt 1 ]; then
    only_source="--only-source"
fi

fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --srcdict ./Data/it-mono/dict.$SRC.txt \
    --tgtdict ./Data/it-mono/dict.$TGT.txt \
    --trainpref $train_file.tok.spm \
        --validpref $dev_file.tok.spm \
        --testpref $test_file.tok.spm \
    --destdir "$(dirname $train_file)/bin" \
        --thresholdtgt 0 --thresholdsrc 0 --workers 20 $only_source

# NOTE: if monolingual, --only-source
# repeat in opposite direction if required, binary files are directional