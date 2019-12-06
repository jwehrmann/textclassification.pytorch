import torch
from data import get_loaders
from train import train, test, check_input
import models 
from matplotlib import pyplot as plt


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
batch_size = 32
device_name = 'cuda'
nb_epochs = 3
log_interval = 100
lr = 1e-3
nb_epochs = 10

device = torch.device(device_name)

train_loader, valid_loader = get_loaders(
    data_path=data_path, 
    batch_size=batch_size, 
    splits=['train', 'valid'],
)

nb_words = len(train_loader.dataset.vocab)

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

model = models.TextLSTM()
model = model.to(device)

dummy_pred = check_input(model, device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,)

history = train(
    model=model, train_loader=train_loader, 
    test_loader=valid_loader, device=device, optimizer=optimizer, 
    lr_scheduler=lr_scheduler, nb_epochs=4, 
    log_interval=100
)
print('Max val acc: {:.2f}%'.format(max(history['val_acc'])))

test_loader = get_loaders(
    data_path=data_path, 
    batch_size=batch_size, 
    splits=['test'],
)[0]
