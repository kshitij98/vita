#!/bin/bash

DATADIR=$1

echo "DATADIR = $DATADIR"

SPM=spm_encode
MODEL=${DATADIR}/mbart.cc25.v2/sentence.bpe.model
DATA=${DATADIR}/preprocessed
SRC=en_XX
TGT=hi_IN

python3 scripts/prepare_data.py $DATADIR

for Item in 'train' 'train_mask' 'valid' 'test' 'challenge-test-set';
	do
		${SPM} --model=${MODEL} < ${DATA}/${Item}.${SRC} > ${DATA}/${Item}.spm.${SRC}
		${SPM} --model=${MODEL} < ${DATA}/${Item}.${TGT} > ${DATA}/${Item}.spm.${TGT}
	done

python3 scripts/fix_mask_token.py $DATADIR/preprocessed/train_mask.spm.en_XX

if [ ! -d "$DATADIR/trimmed" ]; then
    echo "Creating directory - BASE_DIR/trimmed"
    mkdir $DATADIR/trimmed -p
fi

python3 scripts/build_vocab.py --corpus-data "$DATADIR/preprocessed/*.spm.*" --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output "$DATADIR/trimmed/dict.txt"
python3 scripts/trim_mbart.py --pre-train-dir "$DATADIR/mbart.cc25.v2/" --ft-dict "$DATADIR/trimmed/dict.txt" --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output "$DATADIR/trimmed/model.pt"