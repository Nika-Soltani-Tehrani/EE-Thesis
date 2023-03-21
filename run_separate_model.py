import time
from models.Separate_model import SeparateModel
from data import create_dataset
import torch
import torchvision
import torchvision.transforms as transforms
from configs.config_train import cfg

# Create dataloaders
if cfg.dataset_mode == 'CIFAR10':
    print('here')
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomCrop(32, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=cfg.dataroot, train=True,
                                            download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif cfg.dataset_mode == 'CIFAR100':
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomCrop(32, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root=cfg.dataroot, train=True,
                                             download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif cfg.dataset_mode == 'CelebA':
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

elif cfg.dataset_mode == 'OpenImage':
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

else:
    raise Exception('Not implemented yet')

model = SeparateModel()
# Train with the Discriminator
for i, data in enumerate(dataset):  # loop over all images

    if cfg.dataset_mode in ['CIFAR10', 'CIFAR100']:
        print(data[0].shape)
        input_bit_stream = data[0]
    elif cfg.dataset_mode == 'CelebA':
        input_bit_stream = data['data']
    elif cfg.dataset_mode == 'OpenImage':
        input_bit_stream = data['data']

    model.set_input(input_bit_stream)         # unpack data from dataset and apply preprocessing
    model.execute()


# import numpy as np
# from pyldpc import decode, encode, make_ldpc, get_message
#
# n =15
# d_v = 4
# d_c = 5
# snr = 20
# H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
# print(H)
# print(G)
# k = G.shape[1]
# v = np.random.randint(2, size=k)  # main signal
# y = encode(G, v, snr)  # encoded signal
# d = decode(H, y, snr)  # decoded signal, 15 bits as n = 15
# print(y)
# print(d)
# x = get_message(G, d)  # main message (without parity)
# print(x)
# print(abs(x - v).sum())
