#!/bin/bash

DATADIR=$1
PRETRAIN=$2
DICT=$3

TRAIN=train_mask
VALID=valid
TEST=test
CHALLENGE=challenge-test-set

echo "Using ${TRAIN} for masked training"

DATA=${DATADIR}/preprocessed
SRC=en_XX
TGT=hi_IN
NAME=en-hi
DEST=${DATADIR}/postprocessed/mask

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
POSTPROCESSED=${DATADIR}/postprocessed/mask/${NAME}
SAVEDIR=${DATADIR}/checkpoint/mask

WARMUP=400 STEPS=8500 SAVE_FREQ=350

fairseq-preprocess \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}.spm \
--validpref ${DATA}/${VALID}.spm \
--testpref ${DATA}/${TEST}.spm  \
--destdir ${DEST}/${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70

fairseq-train ${POSTPROCESSED}  --encoder-normalize-before --decoder-normalize-before \
 --arch mbart_large --task translation_from_pretrained_bart  --source-lang ${SRC} --target-lang ${TGT} \
 --criterion label_smoothed_cross_entropy --label-smoothing 0.2  --dataset-impl mmap --optimizer adam \
 --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 1e-05 --min-lr -1 \
 --warmup-updates $WARMUP --max-update $STEPS --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
 --max-tokens 512 --update-freq 2 --save-interval 1 --save-interval-updates $SAVE_FREQ --keep-interval-updates 10 \
 --seed 222 --log-format simple --log-interval 2 --reset-optimizer --reset-meters \
 --reset-dataloader --reset-lr-scheduler --restore-file $PRETRAIN --langs $langs --layernorm-embedding  \
 --ddp-backend no_c10d --save-dir ${SAVEDIR}
