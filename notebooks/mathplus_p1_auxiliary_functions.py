__version__ = '0.1.1'

import numpy as np

# import engine module
import sirf.Gadgetron as mr

# import further modules
import os
from numpy.lib.stride_tricks import as_strided

import matplotlib.pyplot as plt

import numpy as np 


def crop_and_fill(templ_im, vol):
    """Crop volumetric image data and replace image content in template image object"""
    # Get size of template image and crop
    idim_orig = templ_im.as_array().shape
    idim = (1,)*(3-len(idim_orig)) + idim_orig
    offset = (np.array(vol.shape) - np.array(idim)) // 2
    vol = vol[offset[0]:offset[0]+idim[0], offset[1]:offset[1]+idim[1], offset[2]:offset[2]+idim[2]]
    
    # Make a copy of the template to ensure we do not overwrite it
    templ_im_out = templ_im.copy()
    
    # Fill image content 
    templ_im_out.fill(np.reshape(vol, idim_orig))
    return(templ_im_out)


'''
Variable density Cartesian sampling taken from
https://github.com/js3611/Deep-MRI-Reconstruction/blob/master/utils/compressed_sensing.py
'''

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, acc, sample_n=10):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    return mask


def undersample_cartesian_data(ad: pMR.AcquisitionData):

    ky_index = ad.get_ISMRMRD_info('kspace_encode_step_1')
    cph_index = ad.get_ISMRMRD_info('phase')
    ky_num = int(np.max(ky_index)+1)
    cph_num = int(np.max(cph_index)+1)
    print(f'Nky {ky_num} - Ncph {cph_num}')

    R = 4
    F = int(ky_num/10)
    msk = cartesian_mask([cph_num, ky_num, 1], R, sample_n=F)


    acq_us = ad.new_acquisition_data(empty=True)

    # Create raw data
    for cnd in range(cph_num):
        for ynd in range(ky_num):
            if msk[cnd, ynd, 0] == 1:
                cidx = np.where((ky_index == ynd) & (cph_index == cnd))[0]
                if len(cidx) > 0:
                    cacq = ad.acquisition(cidx)
                    acq_us.append_acquisition(cacq)
                else:
                    print(f'ky {ynd} - cph {cnd} not found')

    acq_us.sort()     

    return acq_us