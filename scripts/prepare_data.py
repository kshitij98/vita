from tqdm import tqdm
import os
import sys
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
import json
import pandas as pd
import random
import copy

DATADIR = sys.argv[1]

TAGS = False
if not os.path.exists(f'{DATADIR}/preprocessed'):
    os.makedirs(f'{DATADIR}/preprocessed')

split_names = {
    'train': 'train',
    'dev': 'valid',
    'test': 'test',
    'challenge-test-set': 'challenge-test-set',
}
splits = ['train', 'dev', 'test', 'challenge-test-set']

MAX_TAGS = 10

def create_masked_samples(sentence, samples=5, num_masks=1, mask='<mask>'):
    ans = []
    words = sentence.strip().split(' ')
    num_words = len(words)
    for i in range(samples):
        ids = list(range(num_words))
        random.shuffle(ids)
        ids = ids[:num_masks]
        masked_words = copy.deepcopy(words)
        for word_id in ids:
            masked_words[word_id] = mask
        ans.append(' '.join(masked_words))
    
    return ans

# Parse Hindi Visual Genome Dataset
def hindi_genome(splits=['train', 'dev', 'test', 'challenge-test-set'], split_names={}, use_tags=False):
    normalizer = DevanagariNormalizer()
    data = {}

    for split in splits:
        data[f'{split_names[split]}.en_XX'] = []
        data[f'{split_names[split]}.hi_IN'] = []
    data['train_mask.en_XX'] = []
    data['train_mask.hi_IN'] = []

    if use_tags:
        with open(f'{DATADIR}/hindi-genome/object_tags.json') as f:
            object_tags = json.load(f)

    raw_name = f'{DATADIR}/hindi-genome/hindi-visual-genome-11/hindi-visual-genome-<split>.txt'
    for split in splits:
        name = raw_name.replace('<split>', split)
        df = pd.read_csv(name, delimiter='\t', encoding='utf-8', header=None)
        for i in tqdm(range(len(df))):
            english = df.iloc[i, 5]
            hindi = df.iloc[i, 6]
            image_id = df.iloc[i, 0]

            if use_tags:
                tags = []
                for label in object_tags[image_id]:
                    if label in tags:
                        continue
                    tags.append(label)
                
                if len(tags) > MAX_TAGS:
                    tags = tags[:MAX_TAGS]
                tag_str = ', '.join(tags)

            if split == 'train':
                random.seed(42)
                masked_sentences = create_masked_samples(english)
                for sentence in masked_sentences:
                    if use_tags:
                       data['train_mask.en_XX'].append(sentence.strip() + ' ## ' + tag_str + '\n')
                    else:
                       data['train_mask.en_XX'].append(sentence.strip() + '\n')

                    data['train_mask.hi_IN'].append(hindi.strip() + '\n')

            
            if use_tags:
                data[f'{split_names[split]}.en_XX'].append(english.strip() + ' ## ' + tag_str + '\n')
            else:
                data[f'{split_names[split]}.en_XX'].append(english.strip() + '\n')

            data[f'{split_names[split]}.hi_IN'].append(hindi.strip() + '\n')

    for key in data.keys():
        if key.endswith('.hi_IN'):
            for i in range(len(data[key])):
                data[key][i] = normalizer.normalize(data[key][i])

    return data

file_mapping = hindi_genome(splits, split_names, use_tags=False)

print(file_mapping.keys())

for k, v in file_mapping.items():
    print(f'total size of {k} data is {len(v)}')
    with open(f'{DATADIR}/preprocessed/{k}', 'w') as fp:
        fp.writelines(v)
