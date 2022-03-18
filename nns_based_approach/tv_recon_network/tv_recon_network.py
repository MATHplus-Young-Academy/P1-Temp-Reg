#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..operators.grad_operators import GradOperators
from cg import CG


class LearnedTVMapCNN(nn.Module):

    """
    An implementation of a network which solves the following problem
        min_{x,z} 1/2 * ||Ax - y||_2^2 + || \Lambda*z ||_1 + \beta/2 ||Gx - z||_2^2, (1)

    assuming that \Lambda and \beta are fixed.

    The parameter beta can be either chosen a-priori or trained as well.

    The parameter map \Lambda is estimated within the network.
    inputs:
        - CNN_block ... CNN-block used to estimate \Lambda
        - operator .. object which contains forward and adjoint of the acquisition model
        - T ... length of the network
    """

    def __init__(self, CNN_block, T=8, beta_reg=10.0):

        super(LearnedTVMapCNN, self).__init__()

        self.T = T
        self.CNN_block = CNN_block
        self.GOps = GradOperators
        self.beta_reg = torch.as_tensor(beta_reg)  # can be any real number (use torch.exp to make positive) #TODO: make Parameter

    @staticmethod
    def apply_soft_threshold(x, threshold):
        # the soft-thresholding can be expressed as
        # S_t(x) = ReLU(x-t) - ReLU(-x -t)
        return F.relu(x - threshold) - F.relu(-x - threshold)

    def solve_S2(self, Lambda_map, x):
        # sub-problem 2: solve (1) with respect to z, for fixed x, i.e.
        # z* = sof-threhsolding(Gx)

        threshold = Lambda_map / torch.exp(self.beta_reg)
        Gx = self.GOps.apply_G(x)  # obtain z by sof-thresholding Gx (component-wise)
        z = self.apply_soft_threshold(self, Gx, threshold)
        return z

    def solve_S1(self, acq_model, z, y):
        # sub-problem 1: solve (1) with respect to x, i.e. solve Hx=b, with
        # H = A^H A + beta*G^H G
        # b = A^H + beta*z

        return CG.apply(z, acq_model, self.beta, y, G=self.GOps.apply_G, GH=self.GOps.apply_GH, GHG=self.GOps.apply_GHG)

    def forward(self, y, acq_model):
        x_sirf = acq_model.adjoint(y)
        x = torch.as_tensor(x_sirf.as_array())
        x_device = x.to(self.CNN_block.device)

        # obtain Lambda as output of the CNN-block
        Lambda_map = self.CNN_block(x_device)  # has three channels (for x-,y- and t-dimension)
        Lambda_map = torch.exp(Lambda_map)
        Lambda_map = torch.cat(2 * [lambda_reg], dim=1)  # will have six channels (for real and imaginary part)
        Lambda_map = Lambda_map.cpu()

        for kiter in range(self.T):
            z = self.solve_S2(Lambda_map, x)
            x = self.solve_S1(acq_model, z, y)
        return x, Lambda_map
