#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn

import numpy as np

import sys
sys.path.append('../operators/')


from operators.grad_operators import GradOperators

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
    
    def __init__(self, CNN_block, T=8, beta_reg=10.):
        
        super(LearnedTVMapCNN, self).__init__()
        
        self.T = T
        self.CNN_block = CNN_block
        self.GOps = GradOperators
        self.beta_reg = beta_reg #can be any real number (use torch.exp to make positive)
        
    def apply_A(self, x, acq_model):
        
        return 0.
        
    def apply_AH(self, y, acq_model):
        
        return 0.
        
    def apply_soft_threshold(self, z, threshold):
        #the soft-thresholding can be expressed as 
        #S_t(x) = ReLU(x-t) - ReLU(-x -t)
        return nn.ReLU()(x-threshold) - nn.ReLU()(-x -threshold)

    def forward(self, x, acq_model):
        
        #get sizes and device
        mb,n_ch,Nx,Ny,Nt = x.shape
        
        #obtain Lambda as output of the CNN-block
        Lambda_map = self.CNN_block(x) #has three channels (for x-,y- and t-dimension)
        Lambda_map = torch.exp(Lambda_map)
        Lambda_map = torch.cat(2*[lambda_reg],dim=1) #will have six channels (for real and imaginary part)
        
        for kiter in range(self.T):
            
            #sub-problem 2: solve (1) with respect to z, for fixed x, i.e.
            # z* = sof-threhsolding(Gx)
            threshold = Lambda_map / torch.exp(self.beta_reg)
            
            #obtain z by sof-thresholding Gx (component-wise)
            Gx = self.GOps.apply_G(x)
            z = self.apply_soft_threshold(self, Gx, threshold)
            
            #sub-problem 1: solve (1) with respect to x, i.e. solve Hx=b, with
            #H = A^H A + beta*G^H G
            #b = A^H + beta*z
            x = 0.
        
        return x, Lambda_map
