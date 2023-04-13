import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim

import util
from configs.config_train import cfg
from . import channel

# class SeparateModel:
#
#     def __init__(self):
#         # Set the path to the CIFAR-10 dataset directory
#         self.data_dir = '/path/to/cifar10/dataset'
#         # Set the path to the encoded and LDPC-coded CIFAR-10 dataset directory
#         self.encoded_dir = 'data/cifar10/encoded'
#         self.ldpc_dir = 'data/cifar10/ldpc'
#         # Define the BPG encoding and decoding commands
#         self.bpg_enc_cmd = 'bpgenc -o {output_path} {input_path}'
#         self.bpg_dec_cmd = 'bpgdec {input_path} -o {output_path}'
#         self.SNR = 5
#         # Define the LDPC code parameters
#         self.n = 2400  # codeword length
#         self.k = 1200  # message length
#         self.rate = self.k / self.n  # code rate
#         # Define the QAM modulation parameters
#         self.n_bits = 8  # number of bits per symbol
#         self.n_symbols = 256  # number of symbols
#         self.constellation = np.linspace(-1, 1, self.n_symbols)  # define the constellation
#         self.gpu_ids = []
#         self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
#
#     @staticmethod
#     def name():
#         return 'Separate_Model'
#
#     def set_input(self, image):
#         self.real_A = image.clone().to(self.device)
#
#
#     def set_img_path(self, path):
#         self.image_paths = path
#
#     def execute(self, opt):
#         image = self.real_A
#
#         # Save the image to a temporary file
#         # print(image.shape)
#         temp_image_path = 'temp.png'
#         imageio.imwrite(temp_image_path, image)
#
#         # Apply BPG encoding to the image
#         encoded_path = os.path.join(self.encoded_dir, file + '.bpg')
#         os.system(self.bpg_enc_cmd.format(input_path=temp_image_path, output_path=encoded_path))
#
#         # Apply LDPC coding to the encoded message
#         encoded_message = np.fromfile(encoded_path, dtype=np.uint8)
#         ldpc_path = os.path.join(self.ldpc_dir, file + '.ldpc')
#         np.random.seed(0)  # for reproducibility
#         ldpc_matrix = np.random.randint(0, 2, size=(self.n - self.k, self.n))
#         ldpc_coded_message = np.zeros(self.n, dtype=np.uint8)
#         ldpc_coded_message[:self.k] = encoded_message
#         ldpc_coded_message[self.k:] = np.dot(ldpc_matrix, encoded_message) % 2
#         ldpc_coded_message.tofile(ldpc_path)
#
#         # Convert the LDPC-coded message to a sequence of QAM symbols
#         qam_symbols = np.zeros(ldpc_coded_message.size // self.n_bits, dtype=complex)
#         for i in range(qam_symbols.size):
#             symbol = 0
#             for j in range(self.n_bits):
#                 symbol += ldpc_coded_message[i * self.n_bits + j] * 2 ** (self.n_bits - j - 1)
#             qam_symbols[i] = self.constellation[symbol]
#
#         # Plot the QAM constellation
#         plt.figure()
#         plt.scatter(qam_symbols.real, qam_symbols.imag)
#         plt.xlim((-1.5, 1.5))
#         plt.ylim((-1.5, 1.5))
#         plt.title('QAM Constellation')
#         plt.show()
#         from . import channel
#         channel = channel.WOOFDMChannel(opt, self.device, pwr=1)
#         out_pilot, out_sig, H_true, noise_pwr, PAPR = channel(qam_symbols, SNR=self.SNR, cof=None)




# Set the path to the CIFAR-10 dataset directory
data_dir = 'data/cifar10-separate'
# Set the path to the encoded and LDPC-coded CIFAR-10 dataset directory
encoded_dir = 'data/cifar10/encoded'
ldpc_dir = 'data/cifar10/ldpc'
# Define the BPG encoding and decoding commands
bpg_enc_cmd = 'bpgenc -o {output_path} {input_path}'
bpg_dec_cmd = 'bpgdec {input_path} -o {output_path}'
# Define the LDPC code parameters
n = 2400  # codeword length
k = 1200  # message length
SNR = 5
rate = k/n  # code rate
# Define the QAM modulation parameters
n_bits = 8  # number of bits per symbol
n_symbols = 256  # number of symbols
constellation = np.linspace(-1, 1, n_symbols)  # define the constellation

PSNR_list = []
SSIM_list = []
MSSSIM_list = []
H_err_MMSE_list = []
H_err_list = []
PAPR_list = []


# Loop over all CIFAR-10 images in the directory
for root, dirs, files in os.walk(data_dir):
    for file in files:
        # Load the image and convert it to a numpy array
        image_path = os.path.join(root, file)
        image = imageio.imread(image_path)
        image = image / 255.0

        # Save the image to a temporary file
        temp_image_path = './temp.png'
        imageio.imwrite(temp_image_path, image)

        # Apply BPG encoding to the image
        encoded_path = os.path.join(encoded_dir, file)
        imageio.imwrite(encoded_path, image)
        os.system(bpg_enc_cmd.format(input_path=temp_image_path, output_path=encoded_path+'.bpg'))

        # Apply LDPC coding to the encoded message
        encoded_message = np.fromfile(encoded_path, dtype=np.uint8)
        # encoded_message = encoded_message[:k]

        ldpc_path = os.path.join(ldpc_dir, file + '.ldpc')
        np.random.seed(0)  # for reproducibility
        ldpc_matrix = np.random.randint(0, 2, size=(n-k, n))
        ldpc_coded_message = np.zeros(n, dtype=np.uint8)
        print(encoded_message)
        ldpc_coded_message[:k] = encoded_message
        ldpc_coded_message[k:] = np.dot(ldpc_matrix, encoded_message) % 2
        ldpc_coded_message.tofile(ldpc_path)
        # Convert the LDPC-coded message to a sequence of QAM symbols
        qam_symbols = np.zeros(ldpc_coded_message.size // n_bits, dtype=complex)
        for i in range(qam_symbols.size):
            symbol = 0
            for j in range(n_bits):
                symbol += ldpc_coded_message[i * n_bits + j] * 2**(n_bits - j - 1)
            qam_symbols[i] = constellation[symbol]
        # Plot the QAM constellation
        plt.figure()
        plt.scatter(qam_symbols.real, qam_symbols.imag)
        plt.xlim((-1.5, 1.5))
        plt.ylim((-1.5, 1.5))
        plt.title('QAM Constellation')
        plt.show()

        channel = channel.WOOFDMChannel(cfg, device=None, pwr=1)
        out_pilot, out_sig, H_true, noise_pwr, PAPR = channel(qam_symbols, SNR=SNR, cof=None)

# Get the int8 generated images
    img_gen_numpy = out_sig.detach().cpu().float().numpy()
    img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8)

    origin_numpy = image.detach().cpu().float().numpy()
    origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    origin_int8 = origin_numpy.astype(np.uint8)

    diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8)) ** 2, (1, 2, 3))

    PSNR = 10 * np.log10((255 ** 2) / diff)
    PSNR_list.append(np.mean(PSNR))

    img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
    origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

    ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(cfg.how_many_channel, 1, 1, 1), data_range=255,
                    size_average=False)  # return (N,)
    SSIM_list.append(torch.mean(ssim_val).item())

    ms_ssim_val = ms_ssim(img_gen_tensor, origin_tensor.repeat(cfg.how_many_channel, 1, 1, 1), data_range=255, size_average=False)  # (N,)
    MSSSIM_list.append(torch.mean(ms_ssim_val).item())

    # Save the first sampled image
    save_path = f'{cfg.image_out_path}/{i}_PSNR_{PSNR[0]:.3f}_SSIM_{ssim_val[0]:.3f}_MS-SSIM_{ms_ssim_val[0]:.3f}.png'
    util.save_image(util.tensor2im(out_sig[0].unsqueeze(0)), save_path, aspect_ratio=1)

    save_path = f'{cfg.image_out_path}/{i}.png'
    util.save_image(util.tensor2im(input), save_path, aspect_ratio=1)

    if i % 10 == 0:
        print(i)

print(f'PSNR: {np.mean(PSNR_list)}')
print(f'SSIM: {np.mean(SSIM_list)}')
print(f'MS-SSIM: {np.mean(MSSSIM_list)}')
print(f'CE refined: {np.mean(H_err_list)}')
print(f'CE MMSE: {np.mean(H_err_MMSE_list)}')
print(f'PAPR: {np.mean(PAPR_list)}')
