# ViTA

## Installation

```bash
conda env create -f environment.yml
conda activate vita
python -m spacy download en_core_web_sm
```

## Downloads

```bash
export DATADIR=<path to data directory>

# Download mBART
cd $DATADIR
wget -c https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
tar -xzvf mbart.cc25.v2.tar.gz

# Download Hindi Visual Genome dataset
mkdir -p $DATADIR/hindi-genome
cd $DATADIR/hindi-genome
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3267/hindi-visual-genome-11.zip
unzip hindi-visual-genome-11.zip
```

<!-- Download Hindi Visual Genome Dataset -->

## Feature Extraction

The object tags were detected using Faster R-CNN checkpoint with Resnet-101 C4 backbone available in [detectron2](https://github.com/facebookresearch/detectron2)

Store the object tags in a json file as follows:
```json
{
    "<image_id>": ["dog", "cat", "person"],
    ...
}
```

## Data Preprocessing

Preprocess the dataset using SPM and trim mBART vocabulary using the dataset.

```bash
bash preprocess.sh $DATADIR
```

## Finetune

```bash
DICT=$DATADIR/trimmed/dict.txt

# Finetune ViTA with masking
CHECKPOINT=$DATADIR/trimmed/model.pt
bash train_mask.sh $DATADIR $CHECKPOINT $DICT

# Finetune ViTA model
CHECKPOINT=$DATADIR/checkpoint/mask/checkpoint_best.pt
bash train_main.sh $DATADIR $CHECKPOINT $DICT
```
