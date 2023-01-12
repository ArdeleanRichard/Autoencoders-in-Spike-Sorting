from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot      import close, figure, imshow, savefig, show, title
from matplotlib.lines       import Line2D
from os.path                import join
from random                 import sample
from torch import device, no_grad, nn
from torch.nn               import Linear, Module, MSELoss, ReLU, Sequential, Sigmoid, Tanh
from torchvision.utils      import make_grid


class AutoEncoder(Module):
    '''A class that implements an AutoEncoder
    '''
    @staticmethod
    def get_non_linearity(params):
        '''Determine which non linearity is to be used for both encoder and decoder'''
        def get_one(param):
            '''Determine which non linearity is to be used for either encoder or decoder'''
            param = param.lower()
            if param=='relu': return ReLU()
            if param=='sigmoid': return Sigmoid()
            if param=='tanh': return Tanh()
            return None

        decoder_non_linearity = get_one(params[0])
        # encoder_non_linearity = getnl(params[a]) if len(params)>1 else decoder_non_linearity
        encoder_non_linearity = get_one(params[0])

        return encoder_non_linearity, decoder_non_linearity

    @staticmethod
    def build_layer(sizes,
                    non_linearity = None):
        '''Construct encoder or decoder as a Sequential of Linear labels, with or without non-linearities

        Positional arguments:
            sizes   List of sizes for each Linear Layer
        Keyword arguments:
            non_linearity  Object used to introduce non-linearity between layers
        '''
        linears = [Linear(m,n) for m,n in zip(sizes[:-1],sizes[1:])]

        for id, layer in enumerate(linears):
            if id != len(linears) - 1:
                # print(layer.weight)
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.zeros_(layer.weight)
                # print(layer.weight)
                # print(np.mean(layer.weight), np.std(layer.weight))
            else:
                # print(layer.weight)
                # nn.init.xavier_uniform_(layer.weight)
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                # nn.init.zeros_(layer.weight)
                # print(layer.weight)

        if non_linearity==None:
            return Sequential(*linears)
        else:
            return Sequential(*[item for pair in [(layer,non_linearity) if id != len(linears)-1 else (layer,Tanh()) for id, layer in enumerate(linears)] for item in pair])

    def __init__(self,
                 encoder_sizes         = [70,60,50,40,30,20,10,5],
                 encoder_non_linearity = ReLU(inplace=True),
                 decoder_sizes         = [],
                 decoder_non_linearity = ReLU(inplace=True)):
        '''
        Keyword arguments:
            encoder_sizes            List of sizes for each Linear Layer in encoder
            encoder_non_linearity    Object used to introduce non-linearity between encoder layers
            decoder_sizes            List of sizes for each Linear Layer in decoder
            decoder_non_linearity    Object used to introduce non-linearity between decoder layers
        '''
        super().__init__()
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = encoder_sizes[::-1] if len(decoder_sizes)==0 else decoder_sizes


        self.encoder = AutoEncoder.build_layer(self.encoder_sizes,
                                               non_linearity = encoder_non_linearity)
        self.decoder = AutoEncoder.build_layer(self.decoder_sizes,
                                               non_linearity = decoder_non_linearity)

        self.encode  = True
        self.decode  = True


    def forward(self, x):
        '''Propagate value through network

           Computation is controlled by self.encode and self.decode
        '''
        if self.encode:
            x = self.encoder(x)

        if self.decode:
            x = self.decoder(x)
        return x

    def n_encoded(self):
        return self.encoder_sizes[-1]


def train(loader,model,optimizer,criterion, data_dim,
          N   = 25,
          dev = 'cpu',
          verbose=1):
    '''Train network

       Parameters:
           loader       Used to get data
           model        Model to be trained
           optimizer    Used to minimze errors
           criterion    Used to compute errors
      Keyword parameters:
          N             Number of epochs
          dev           Device - cpu or cuda
    '''
    Losses        = []

    for epoch in range(N):
        loss = 0
        for batch_features, _ in loader:
            batch_features = batch_features.view(-1, data_dim).to(dev)
            optimizer.zero_grad()
            outputs        = model(batch_features)
            train_loss     = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        Losses.append(loss / len(loader))
        if verbose == 1:
            print(f'epoch : {epoch+1}/{N}, loss = {Losses[-1]:.6f}')

    return Losses


def reconstruct(loader,model,criterion,device,
                N        = 25,
                prefix   = 'test',
                show     = False,
                figs     = './figs',
                n_images = -1):
    '''Reconstruct images from encoding

       Parameters:
           loader
           model
       Keyword Parameters:
           N        Number of epochs used for training (used in image title only)
           prefix   Prefix file names with this string
           show     Used to display images
           figs     Directory for storing images
    '''

    def plot(original=None,decoded=None):
        '''Plot original images and decoded images'''
        fig = figure(figsize=(10,10))
        ax    = fig.subplots(nrows=2)
        ax[0].imshow(make_grid(original.view(-1,79)).permute(1, 2, 0))
        ax[0].set_title('Raw images')
        scaled_decoded = decoded/decoded.max()
        ax[1].imshow(make_grid(scaled_decoded.view(-1,79)).permute(1, 2, 0))
        ax[1].set_title(f'Reconstructed images after {N} epochs')
        savefig(join(figs,f'{prefix}-comparison-{i}'))
        if not show:
            close (fig)

    samples = [] if n_images==-1 else sample(range(len(loader)//loader.batch_size), k = n_images)
    loss = 0.0
    with no_grad():
        for i,(batch_features, _) in enumerate(loader):
            batch_features = batch_features.view(-1, 784).to(device)
            outputs        = model(batch_features)
            test_loss      = criterion(outputs, batch_features)
            loss          += test_loss.item()
            if len(samples)==0 or i in samples:
                plot(original=batch_features,
                    decoded=outputs)


    return loss


def plot_losses(Losses,
                lr                   = 0.001,
                encoder              = [],
                decoder              = [],
                encoder_nonlinearity = None,
                decoder_nonlinearity = None,
                N                    = 25,
                show                 = False,
                figs                 = './figs',
                prefix               = 'ae',
                test_loss            = 0):
    '''Plot curve of training losses'''
    fig = figure(figsize=(10,10))
    ax  = fig.subplots()
    ax.plot(Losses)
    ax.set_ylim(bottom=0)
    ax.set_title(f'Training Losses after {N} epochs')
    ax.set_ylabel('MSELoss')
    ax.text(0.95, 0.95, '\n'.join([f'lr = {lr}',
                                   f'encoder = {encoder}',
                                   f'decoder = {decoder}',
                                   f'encoder nonlinearity = {encoder_nonlinearity}',
                                   f'decoder nonlinearity = {decoder_nonlinearity}',
                                   f'test loss = {test_loss:.3f}'
                                   ]),
            transform           = ax.transAxes,
            fontsize            = 14,
            verticalalignment   = 'top',
            horizontalalignment = 'right',
            bbox                = dict(boxstyle  = 'round',
                                       facecolor = 'wheat',
                                       alpha     = 0.5))
    savefig(join(figs,f'{prefix}-losses'))
    if not show:
        close (fig)



def plot_encoding(loader,model,
                figs    = './figs',
                dev     = 'cpu',
                colours = [],
                show    = False,
                prefix  = 'ae'):
    '''Plot the encoding layer

       Since this is multi,dimensional, we will break it into 2D plots
    '''
    def extract_batch(batch_features, labels):
        '''Extract xs, ys, and colours for one batch'''

        batch_features = batch_features.view(-1, 79).to(dev)
        encoded        = model(batch_features).tolist()
        return list(zip(*([encoded[k][0] for k in range(len(labels))],
                          [encoded[k][1] for k in range(len(labels))],
                          [labels[k] for k in range(len(labels))],
                          [colours[int(labels.tolist()[k])] for k in range(len(labels))])))

    save_decode  = model.decode
    model.decode = False
    with no_grad():
        fig     = figure(figsize=(10,10))

        xs, ys, ls, cs = tuple(zip(*[xylc for batch_features, labels in loader for xylc in extract_batch(batch_features, labels)]))

        plt.scatter(xs,ys,c=cs,s=1)


    savefig(join(figs,f'{prefix}-encoding'))
    if not show:
        close(fig)

    model.decode = save_decode



def get_latent_code(loader, model, dev='cpu'):
        '''Plot the encoding layer

           Since this is multi,dimensional, we will break it into 2D plots
        '''

        def extract_batch(batch_features, labels):
            '''Extract xs, ys, and colours for one batch'''

            batch_features = batch_features.view(-1, 79).to(dev)
            encoded = model(batch_features).tolist()
            return list(zip(*([encoded[k][0] for k in range(len(labels))],
                              [encoded[k][1] for k in range(len(labels))],
                              [labels[k] for k in range(len(labels))])))

        save_decode = model.decode
        model.decode = False
        with no_grad():
            xs, ys, ls = tuple(
                zip(*[xyc for batch_features, labels in loader for xyc in extract_batch(batch_features, labels)]))

        return np.vstack((xs, ys)).T, ls


