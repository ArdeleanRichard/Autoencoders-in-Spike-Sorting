import os

import numpy as np
import torch
from torch                  import device, no_grad
from torch.cuda             import is_available
from torch.nn               import Linear, Module, MSELoss, ReLU, Sequential, Sigmoid
from torch.optim import Adam, SGD, RMSprop, Rprop
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, ToTensor

from dataset_parsing.simulations_dataset import get_dataset_simulation
from neural_networks.ae_pytorch import AutoEncoder, reconstruct, train, plot_losses, plot_encoding, get_latent_code
from neural_networks.data_iterator import DataIterator
from preprocess.data_scaling import spike_scaling_min_max
from validation.performance import compute_metrics_by_kmeans

os.chdir("../")





def run_ae_pytorch():
    ENCODER = [79, 70, 60, 50, 40, 30, 20, 10, 5, 2]  # sizes of encoder layers
    ENCODER = [79, 40, 20, 10, 2]  # sizes of encoder layers
    DECODER = []  # Decoder layers will be a mirror image of encoder
    LR = 0.001  # Learning rate
    N = 100  # Number of epochs

    # print(is_available())
    dev = device("cuda" if is_available() else "cpu")

    encoder_non_linearity, decoder_non_linearity = AutoEncoder.get_non_linearity(['relu'])
    model = AutoEncoder(encoder_sizes=ENCODER,
                        encoder_non_linearity=encoder_non_linearity,
                        decoder_non_linearity=decoder_non_linearity,
                        decoder_sizes=DECODER).to(dev)
    optimizer = Adam(model.parameters(), lr=LR)
    # optimizer = SGD(model.parameters(), lr=LR, momentum=0) #panarama, toate un punct
    # optimizer = RMSprop(model.parameters(), lr=LR, alpha=0.09, momentum=0.09) #ca adam
    # optimizer = Rprop(model.parameters(), lr=LR) #ca adam
    criterion = MSELoss()
    transform = Compose([ToTensor()])

    # PATH = os.getcwd() + "\muie.pth"
    # state = {'model': model.state_dict()}
    # torch.save(state, PATH)
    # model.load_state_dict(torch.load(PATH)['model'])
    # # print weights
    # for k, v in model.named_parameters():
    #     weights = v[0].detach().numpy()
    #     print(k, np.mean(weights), np.std(weights))

    spikes, labels = get_dataset_simulation(SIM_NR)
    spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes)) * 2 - 1

    tensor_x = torch.Tensor(spikes_scaled)  # transform to torch tensor
    tensor_y = torch.Tensor(labels)  # transform to torch tensor
    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = DataLoader(my_dataset, batch_size=128, shuffle=True)  # create your dataloader

    Losses = train(train_loader,
                   model,
                   optimizer,
                   criterion,
                   data_dim=spikes.shape[-1],
                   N=N,
                   dev=dev, verbose=0)

    # test_loss = reconstruct(train_loader,
    #                         model,
    #                         criterion,
    #                         N        = N,
    #                         show     = True,
    #                         figs     = '.',
    #                         n_images = 5,
    #                         prefix   = 'foo',
    #                         device=dev)

    # plot_losses(Losses,
    #             lr=LR,
    #             encoder=model.encoder_sizes,
    #             decoder=model.decoder_sizes,
    #             encoder_nonlinearity=encoder_non_linearity,
    #             decoder_nonlinearity=decoder_non_linearity,
    #             N=N,
    #             show=True,
    #             figs='.',
    #             prefix='foo',
    #             test_loss=test_loss)

    # plot_encoding(train_loader,
    #               model,
    #               show=True,
    #               colours=['xkcd:purple',
    #                        'xkcd:green',
    #                        'xkcd:blue',
    #                        'xkcd:pink',
    #                        'xkcd:brown',
    #                        'xkcd:red',
    #                        'xkcd:magenta',
    #                        'xkcd:yellow',
    #                        'xkcd:light teal',
    #                        'xkcd:puke'],
    #               figs='.',
    #               prefix='foo')

    test, labels = get_latent_code(train_loader, model)

    return test, labels, model


SIM_NR = 1
metrics = []
RUNS = 100
for i in range(1, RUNS):
    print(i)
    features, gt, model = run_ae_pytorch()
    # scatter_plot.plot(f'Autoencoder on Sim{SIM_NR}', features, gt, marker='o')
    # plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{ae_type}_sim{SIM_NR}_plot{i}')
    met = compute_metrics_by_kmeans(features, gt)
    metrics.append(met)
    print(met)


# np.savetxt(f"./validation_ae/ae_{ae_type}_sim{SIM_NR}_variability_{RUNS}.csv", np.around(a=np.array(metrics).transpose(), decimals=3), delimiter=",")
# np.savetxt(f"./validation_ae/ae_{ae_type}_depth{len(LAYERS)}_sim{SIM_NR}_variability_{RUNS}.csv", metrics, fmt="%.3f", delimiter=",")
np.savetxt(f"./validation_ae/ae_pytorch_sim{SIM_NR}_variability_{RUNS}_init_mod4.csv", metrics, fmt="%.3f", delimiter=",")