#!/bin/bash

DATADIR=$1

wget -c https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz -O $DATADIR/mbart.cc25.v2.tar.gz
tar -xzvf $DATADIR/mbart.cc25.v2.tar.gz -C $DATADIR

mkdir -p $DATADIR/hindi-genome
cd $DATADIR/hindi-genome && { curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3267{/README.txt,/hindi-visual-genome-train.txt.gz,/hindi-visual-genome-dev.txt.gz,/hindi-visual-genome-test.txt.gz,/hindi-visual-genome-challenge-test-set.txt.gz,/hindi-visual-genome-11.zip} ; cd -; }
unzip $DATADIR/hindi-genome/hindi-visual-genome-11.zip -d $DATADIR/hindi-genome
