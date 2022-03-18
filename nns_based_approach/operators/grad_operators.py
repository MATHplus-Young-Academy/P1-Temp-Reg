import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

def pad_circular_nd(x, pad, dim):
    
    """
    function for circular padding
    """

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")


        idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))

        x = torch.cat([x[idx], x], dim=d)
        pass

    return x

def create_grad_kernel(ndims):
    
    """
    function for creating finite-differences kernels
    """

    if ndims == 2:
        dx = np.zeros((3,3,3))
        dy = np.zeros((3,3,3))

        symm_diff = np.array([1,0,-1])

        dx[1,:,1] = symm_diff
        dy[:,1,1] = symm_diff

        filters_list = [dx,dy]
        grad_kernel = torch.zeros(3,1,3,3)
        
    if ndims == 3:
        dx = np.zeros((3,3,3))
        dy = np.zeros((3,3,3))
        dt = np.zeros((3,3,3))

        symm_diff = np.array([1,0,-1])

        dx[1,:,1] = symm_diff
        dy[:,1,1] = symm_diff
        dt[1,1,:] = symm_diff

        filters_list = [dx,dy,dt]
        
        grad_kernel = torch.zeros(3,1,3,3,3)
        for kf in range(ndims):

            h = torch.tensor(filters_list[kf])
            grad_kernel[kf,0,...] = h

    return grad_kernel

class GradOperators(nn.Module):
    
    """
    module which contains 
    """
    
    def __init__(self,dim=3):
        
        super(GradOperators, self).__init__()
        
        self.grad_kernel = create_grad_kernel(dim)
        
    def apply_G(self, x):
        
        #stack kernel to apply it to real and imag part
        grad_kernel = torch.cat(2*[self.grad_kernel],dim=0).to(x.device)
        
        #circular padding
        Gx = F.conv3d( pad_circular_nd(x, 1, [2,3,4]), 
                grad_kernel, 
                bias=None,
                padding=1, #zeropadding cause already circularly padded
                groups=2,
                )
        
        npad=1
        Gx = Gx[...,npad:-npad,npad:-npad,npad:-npad] #crop

        return Gx
    
    def apply_GH(self,z):
        
         #stack kernel to apply it to real and imag part
        grad_kernel = torch.cat(2*[self.grad_kernel],dim=0).to(z.device)
        
        #circular padding
         #circular padding
        GHz = F.conv_transpose3d( pad_circular_nd(z, 1, [2,3,4]), 
                grad_kernel, 
                bias=None,
                padding=1, #zeropadding cause already circularly padded
                groups=2,
                )
        
        npad=1
        GHz = GHz[...,npad:-npad,npad:-npad,npad:-npad]
        
        return GHz
    
    def apply_GHG(self, x):
        
        return self.apply_GH(self.apply_G(x))