#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn

import numpy as np


class LearnedTVMapCNN(nn.Module):
    
    """
    An implementation of a network which solves the following problem
        min_{x,z} 1/2 * ||Ax - y||_2^2 + || \Lambda*z ||_1 + \beta/2 ||Gx - z||_2^2,
    
    assuming that \Lambda and \beta are fixed.
    
    The parameter map \Lambda is estimated within the network.
    inputs: 
        - CNN_block ... CNN-block used to estimate \Lambda
        - operator .. object which contains forward and adjoint of the acquisition model
        - T ... length of the network
    """
    
    
    
	def __init__(self, CNN_block, operator, T=8):
		super(UNet, self).__init__()
        
        
        
        
		