#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import data
from data import get_loaders
from train import train, test, check_input
import models 
from matplotlib import pyplot as plt
import transformers



def plot_instance(instance_id):
    print('\nExample: ')
    print(train_loader.dataset.texts[instance_id])
    print('\nLabel Number: ')
    print(train_loader.dataset.labels[instance_id])
    print('\nLabel String: ')
    print(classes[train_loader.dataset.labels[instance_id]])


classes = [
    'World',
    'Sports',
    'Business',
    'Sci/Tech',
]


data_path = './agnews/'
batch_size = 24
device_name = 'cuda'
nb_epochs = 3
log_interval = 1000
lr = 2e-4
nb_epochs = 10
workers=2

device = torch.device(device_name)

train_dataset = data.TransformersCSVDataset(
    data_path, 'train', 'bert-base-uncased'
)
valid_dataset = data.TransformersCSVDataset(
    data_path, 'valid', 'bert-base-uncased'
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data.collate_fn,
    num_workers=workers,
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=False, 
    collate_fn=data.collate_fn,
    num_workers=workers,
)

nb_words = train_loader.dataset.tokenizer.vocab_size

print(
    'Train size: ', 
    len(train_loader.dataset.texts),
    len(train_loader.dataset.labels)
)
print(
    'Test size : ', 
    len(valid_loader.dataset.texts),
    len(valid_loader.dataset.labels)
)

plot_instance(0)
plot_instance(5000)
plot_instance(1238)
plot_instance(8723)


def freeze_modules(module,):
    for p in module.parameters():
        p.requires_grad = False


def average_pooling(instances, lens):
    return torch.stack([
        text[:l].mean(0) for text, l in zip(instances, lens)
    ])


class TextBERT(nn.Module):
    def __init__(self, num_embeddings=97585, embedding_dim=100):
        '''
            num_embeddings: number of words in the dictionary
            embedding_dim: size of the word-embedding vector
        '''
        super(TextBERT, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        freeze_modules(self.bert)
        self.fc = nn.Linear(768, 4)
        
    def forward(self, x, lengths):
        word_level, sentence_level = self.bert(x)        
        # x = self.fc(sentence_level)
        s = average_pooling(word_level, lengths)
        x = self.fc(s)
        
        return x


model = TextBERT()
model = model.to(device)

dummy_pred = check_input(model, device)

train_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(train_params, lr=lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,)

history = train(
    model=model, train_loader=train_loader, 
    test_loader=valid_loader, device=device, optimizer=optimizer, 
    lr_scheduler=lr_scheduler, nb_epochs=4, 
    log_interval=100
)

print('Max val acc: {:.2f}%'.format(max(history['val_acc'])))
