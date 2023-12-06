import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

##########
import sys
sys.path.append("../../data")
from PTBXLToDataset import CVConditional
from torch.utils.data import DataLoader
##########

import torch.nn as nn
from torch.autograd import Variable
from params import *

#############################
# File Utils
#############################
def get_recursive_files(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        elif file.endswith(ext):
            outFiles.append(os.path.join(folderPath, file))

    return outFiles


def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


#############################
# Plotting utils
#############################
def plot_ecgs(tensor):
    # takes a batch ,n channels , window length and plots the spectogram
    inputs = tensor.detach().cpu().numpy()
    fig, axs = plt.subplots(3, inputs.shape[0], figsize=(18, 50))
    fig.suptitle("Generated I, III and VI leads")
    for i, inp in enumerate(inputs):
        axs[0][i].plot(inp[0])
        axs[1][i].plot(inp[1])
        axs[2][i].plot(inp[2])
    return fig

    # if not (os.path.isdir("visualization")):
    #     os.makedirs("visualization")
    # plt.savefig("visualization/interpolation.png")


def visualize_loss(loss_1, loss_2, first_legend, second_legend, y_label):
    plt.figure(figsize=(10, 5))
    plt.title("{} and {} Loss During Training".format(first_legend, second_legend))
    plt.plot(loss_1, label=first_legend)
    plt.plot(loss_2, label=second_legend)
    plt.xlabel("iterations")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    if not (os.path.isdir("visualization")):
        os.makedirs("visualization")
    plt.savefig("visualization/loss.png")


def latent_space_interpolation(model, logger, global_step, n_samples=5):
    z_test = sample_noise(2)
    with torch.no_grad():
        interpolates = []
        for alpha in np.linspace(0, 1, n_samples):
            interpolate_vec = alpha * z_test[0] + ((1 - alpha) * z_test[1])
            interpolates.append(interpolate_vec)

        interpolates = torch.stack(interpolates)
        generated = model(interpolates)

    logger.add_figure('generated ecgs', plot_ecgs(generated), global_step)


#############################
# Sampling from model
#############################
def sample_noise(size):
    z = torch.FloatTensor(size, noise_latent_dim).to(device)
    z.data.normal_()  # generating latent space based on normal distribution
    return z


#############################
# Model Utils
#############################


def update_optimizer_lr(optimizer, lr, decay):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * decay


def gradients_status(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.bias.data.fill_(0)


#############################
# Creating Data Loader and Sampler
#############################
class WavDataLoader:
    def __init__(self, class_name, input_size, fold_idx, data_dir, sample, equal, smooth, filter, batch_size, dtype="train"):

        if sample:
            stype = "gan_sample"
        elif equal:
            stype = "gan_equal"
        else:
            stype = "gan_no_sample"
            
        self.signals_ds = CVConditional(class_name, input_size // 10, fold_idx, data_dir, type=stype, option=dtype, smooth=smooth, filter=filter, gan=True)
        if dtype == "train":
            self.signals_dl = DataLoader(self.signals_ds, shuffle=True, pin_memory=True,
                                        batch_size=batch_size, num_workers=4, drop_last=True)
        else:
            self.signals_dl = DataLoader(self.signals_ds, shuffle=False, pin_memory=True,
                                        batch_size=batch_size, num_workers=4, drop_last=True)
        self.initialize_iterator()

    def initialize_iterator(self):
        self.data_iter = iter(self.signals_dl)

    def __len__(self):
        return len(self.signals_dl)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.data_iter)
        except StopIteration:
            self.initialize_iterator()
            x = next(self.data_iter)
        return x


if __name__ == "__main__":
    # For debugging purposes
    import time 
    start = time.time()
    print(time.time() - start)
    train_loader = WavDataLoader(os.path.join("piano", "train"), "wav")
    start = time.time()
    for i in range(7):
        x = next(train_loader)
    print(time.time() - start)
