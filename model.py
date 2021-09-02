import torch
import torch.nn as nn
import warnings


class NERF(nn.Module):
    def __init__(self, D, W, L_embed):
        super(NERF, self).__init__()
        layers = []
        last_W = input_W = 3 + 3*2*L_embed

        for i in range(D):
            layers.append(nn.Linear(last_W, W))
            last_W = W
            if i % 4 == 0 and i > 0:
                last_W += input_W

        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(last_W, 4)

        self.act = nn.LeakyReLU(inplace=True)
        warnings.warn('nn.ReLU might cause zero gradient for every parameter, so here I change it to nn.LeakyReLU\n'
                      'use self.act = nn.ReLU(inplace=True) if it can lead to better performance')

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.act(x)
            if i % 4 == 0 and i > 0:
                x = torch.cat([x, input], dim=-1)
        return self.last_layer(x)
