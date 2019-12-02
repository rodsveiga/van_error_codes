
from math import log

import torch
from torch import nn

from utils import default_dtype_torch


class BernoulliMixture(nn.Module):
    def __init__(self, **kwargs):
        super(BernoulliMixture, self).__init__()
        self.n = kwargs['N']
        ### Number of units at each layer
        self.net_width = kwargs['net_width']
        self.z2 = kwargs['z2']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']
        ### Set of weight matrices. There are `net_width` matrices L x L.
        ### Initialized at random.
        self.ber_weight = nn.Parameter(
            torch.rand([self.net_width, self.L, self.L]))
        ### Tensor full of zeros, DIM: number of units at each layer
        self.mix_weight = nn.Parameter(torch.zeros([self.net_width]))

    def forward(self, x):
        raise NotImplementedError

    # sample = +/-1, +1 = up = white, -1 = down = black
    # sample.dtype == default_dtype_torch
    # return sample as dummy of x_hat
    def sample(self, batch_size):
        ### Exponential of the tensor `mix_weight`
        ### Initially: tensor full of zeros -> tensor full of ones
        mix_prob = torch.exp(self.mix_weight)
        ### Sampling from a multinomial distribution with equal prob
        choice = torch.multinomial(mix_prob, batch_size, replacement=True)
        
        ### Elements of the weight matrices are placed in the interval [0, 1] 
        ber_prob = torch.sigmoid(self.ber_weight)
        ### Using the indices sampled from the multinomial distribution, we
        ### choose, based on these indices, a number `net_width` of matrices,
        ### which can be repeated because of `replecement=True`
        ### Note: DIM changes to [batch_size, 1, L, L]
        ### QUESTION: Is it here where the autoregressivity property appears?
        ber_prob = ber_prob[choice, None, :, :]
        ### Then, sampling from a Bernoulli we have a binary sample
        sample = torch.bernoulli(ber_prob).to(default_dtype_torch) * 2 - 1

        return sample, sample


    ### The output `sample` is the input of `_log_prob` function
    def _log_prob(self, sample):
        ### Elements of the weight matrices are placed in the interval [0, 1]
        ber_prob = torch.sigmoid(self.ber_weight)
        ### Note: DIM changes to [1, net_width, 1, L, L]
        ber_prob = ber_prob[None, :, None, :, :]
        ### Mask makes sample go back to {0,1}
        mask = (sample + 1) / 2
        mask = mask[:, None, :, :, :]
        ### Then we can calculate the probability of that specific sample
        ### sample_prob DIM: [batch_size, net_width, 1, L, L ]
        sample_prob = ber_prob * mask + (1 - ber_prob) * (1 - mask)
        ### Is epsilon some kind of noise?
        log_prob = torch.log(sample_prob + self.epsilon)
        ### Reshaping the `log_prob` tensor
        log_prob = log_prob.view(sample.shape[0], self.net_width, -1)
        log_prob = log_prob.sum(dim=2)

        mix_prob = torch.exp(self.mix_weight)
        ### Tensor with uniform entries
        mix_prob = mix_prob / mix_prob.sum()
        ### @: the matrix multiplication is done in the last dimension
        ### shape(log_prob) = [batch_size, net_width]
        ### shape(mix_prob) = [net_width]
        log_prob = torch.log(torch.exp(log_prob) @ mix_prob)
        ### shape(log_prob) [batch_size]

        return log_prob


    ### The output `sample` is the input of `_log_prob` function
    def log_prob(self, sample):
        log_prob = self._log_prob(sample)

        if self.z2:
            # Density estimation on inverted sample
            sample_inv = -sample
            log_prob_inv = self._log_prob(sample_inv)
            log_prob = torch.logsumexp(
                torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob
