# from easydict import EasyDict as edict
from configs.config import cfg

__T = cfg

# Training configs ####################################

if __T.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __T.batch_size = 128
    __T.serial_batches = False  # The batches are continuous or randomly shuffled
    __T.n_epochs = 1  # Number of epochs without lr decay
    __T.n_epochs_decay = 1  # Number of epochs with lr decay
    __T.lr_policy = 'linear'  # decay policy.
    __T.beta1 = 0.5  # parameter for ADAM
    __T.lr = 5e-4  # Initial learning rate
    __T.dataroot = './data'
    __T.size_w = 32
    __T.size_h = 32

elif __T.dataset_mode == 'CelebA':
    __T.batch_size = 64
    __T.serial_batches = False  # The batches are continuous or randomly shuffled
    __T.n_epochs = 15  # Number of epochs without lr decay
    __T.n_epochs_decay = 15  # Number of epochs with lr decay
    __T.lr_policy = 'linear'  # decay policy.  Availability:  see options/train_options.py
    __T.beta1 = 0.5  # parameter for ADAM
    __T.lr = 5e-4  # Initial learning rate
    __T.dataroot = './data/celeba/CelebA_train'
    __T.size_w = 64
    __T.size_h = 64

elif __T.dataset_mode == 'OpenImage':
    __T.batch_size = 24
    __T.serial_batches = False  # The batches are continuous or randomly shuffled
    __T.n_epochs = 10  # Number of epochs without lr decay
    __T.n_epochs_decay = 10  # Number of epochs with lr decay
    __T.lr_policy = 'linear'  # decay policy.  Availability:  see options/train_options.py
    __T.beta1 = 0.5  # parameter for ADAM
    __T.lr = 1e-4  # Initial learning rate
    __T.dataroot = './data/opv6'
    __T.size_w = 256
    __T.size_h = 256

# OFDM configs ####################################

size_latent = (__T.size_w // (2 ** __T.n_downsample)) * (__T.size_h // (2 ** __T.n_downsample)) * (__T.C_channel // 2)
__T.P = 1  # Number of symbols
__T.M = 64  # Number of sub-carriers per symbol
__T.K = 16  # Length of CP
__T.L = 8  # Number of paths
__T.decay = 4  # Exponential decay for the multipath channel
__T.S = size_latent // __T.M  # Number of packets

# Training configs ####################################

__T.print_freq = 100  # frequency of showing training results on console
__T.save_latest_freq = 5000  # frequency of saving the latest results
__T.save_epoch_freq = 1  # frequency of saving checkpoints at the end of epochs
__T.save_by_iter = False  # whether saves model by iteration
__T.continue_train = False  # continue training: load the latest model
__T.epoch_count = 1  # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
__T.verbose = False
__T.isTrain = True
