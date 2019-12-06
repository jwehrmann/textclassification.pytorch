# # Download do dataset
# import os 
# if not os.path.exists('agnews.zip'):
#     !wget https://s3-us-west-2.amazonaws.com/wehrmann/agnews.zip
#     !unzip agnews.zip

import os
import json
import torch
import nltk
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CSVDataset(Dataset):
    """
    """

    def __init__(self, data_path, data_split,):
        self.vocab = load_vocab(os.path.join(data_path, 'vocab.json'))
        self.raw_data = load_txt(
            os.path.join(data_path,'{}.csv'.format(data_split))            
        )
        self.split = data_split

        self.labels, self.texts = get_xy(self.raw_data)
        assert len(self.labels) == len(self.texts)
        self.nb_classes = np.max(self.labels) + 1

    def __getitem__(self, index):
        data = self.texts[index]
        label = self.labels[index]

        tokens = tokenize_text(data, vocab=self.vocab)

        return tokens, label, index, data

    def __len__(self):
        return len(self.labels)
        

class TransformersCSVDataset(Dataset):
    """
    """
    
    def __init__(self, data_path, data_split, pretrained_weights='bert-base-uncased'):
        from transformers import BertTokenizer

        self.raw_data = load_txt(
            os.path.join(data_path,'{}.csv'.format(data_split))            
        )
        self.split = data_split

        self.labels, self.texts = get_xy(self.raw_data)
        assert len(self.labels) == len(self.texts)
        self.nb_classes = np.max(self.labels) + 1
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    def __getitem__(self, index):
        data = self.texts[index]
        label = self.labels[index]

        tokens = torch.tensor(
            self.tokenizer.encode(data), 
            requires_grad=False,
        )

        return tokens, label, index, data

    def __len__(self):
        return len(self.labels)        


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def load_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def load_txt(txt):
    return open(txt).read().strip().split('\n')


def get_xy(raw_data):
    classes = []
    texts = []
    for line in raw_data:
        y, x = line.split('\t')
        y = np.int(y)
        classes.append(y)
        texts.append(x)
    return classes, texts


def tokenize_text(text, vocab, to_tensor=True):
    # Convert caption (string) to word ids.
    tokens = nltk.tokenize.word_tokenize(
        str(text).lower()#.decode('utf-8')
    )
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    if to_tensor:
        caption = torch.Tensor(caption)
    return caption


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, labels, ids, raw = zip(*data)
    labels = torch.Tensor(labels).long()
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    captions, lengths = pad_default(captions)

    return captions, labels, lengths


def pad_default(captions):
    lengths = np.array([len(cap) for cap in captions])
    targets = torch.zeros(len(captions), max(lengths)).long()
    
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return targets, lengths


def get_loaders(
        data_path,
        batch_size,
        workers=2, 
        splits=['train', 'val', 'test'], 
    ):

    loaders = []
    for split in splits:
        csv_dataset = CSVDataset(
            data_path=data_path,
            data_split=split,
        )

        loader = DataLoader(
            dataset=csv_dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),            
            collate_fn=collate_fn,
            num_workers=workers,
        )
        loaders.append(loader)

    return tuple(loaders)