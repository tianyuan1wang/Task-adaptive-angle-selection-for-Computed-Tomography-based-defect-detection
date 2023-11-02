#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:49:17 2023

@author: tianyuan
"""

from scipy import ndimage

import argparse

import skimage
from skimage import draw
import numpy as np

import matplotlib.pyplot as plt

import random

import os

import torch

import astra

import math

import torch.nn as nn
import torch.nn.functional as F


from skimage.metrics import structural_similarity as ssim

# from vqvae import *

# Create a 128x128 NumPy array filled with zeros



# set seed to reproduce

def seed_torch(seed=1029):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True



seed_torch()


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')



def ct_shepp_logan_2d(M, N, r1, r2, ma1, mi1, ma2, mi2, modified=1, E=None, ret_E=None):
    '''Make a 2D phantom.'''

    # Get the ellipse parameters the user asked for
    if E is None:
        if modified:
            E = ct_modified_shepp_logan_params_2d(r1, r2, ma1, mi1, ma2, mi2)
        else:
            E = ct_shepp_logan_params_2d()

    # Extract params
    grey = E[:, 0]
    major = E[:, 1]
    minor = E[:, 2]
    xs = E[:, 3]
    ys = E[:, 4]
    theta = E[:, 5]

    # 2x2 square => FOV = (-1, 1)
    X, Y = np.meshgrid( # meshgrid needs linspace in opposite order
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, M))
    ph = np.zeros((M, N))
    ct = np.cos(theta)
    st = np.sin(theta)

    for ii in range(E.shape[0]):
        xc, yc = xs[ii], ys[ii]
        a, b = major[ii], minor[ii]
        ct0, st0 = ct[ii], st[ii]

        # Find indices falling inside the ellipse
        idx = (
            ((X - xc)*ct0 + (Y - yc)*st0)**2/a**2 +
            ((X - xc)*st0 - (Y - yc)*ct0)**2/b**2 <= 1)

        # Sum of ellipses
        ph[idx] += grey[ii]

    if ret_E:
        return(ph, E)
    return ph


def create_ellipse_mask(h, w, m_l, m_s, scale, h_s, w_s, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2+w_s), int(h/2+h_s))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt(((X - center[0])/m_l)**2 + ((Y-center[1])/m_s)**2)

    mask = dist_from_center <= scale
    return mask.astype(int)




def ct_shepp_logan_params_2d(r1, r2, ma1, mi1, ma2, mi2):
    '''Return parameters for original Shepp-Logan phantom.

    Returns
    -------
    E : array_like, shape (10, 6)
        Parameters for the 10 ellipses used to construct the phantom.
    '''

    E = np.zeros((10, 6)) # (10, [A, a, b, xc, yc, theta])
    E[:, 0] = [2, -.98, -.02, -.02, .01, .01, .01, .01, .01, .01]
    E[:, 1] = [
        .58, .54, ma1, ma2, .21, .046, .046, .046, .023, .023]
    E[:, 2] = [.92, .874, mi1, mi2, .25, .046, .046, .023, .023, .046]
    E[:, 3] = [0, 0, .22, -.22, 0, 0, 0, -.08, 0, .06]
    E[:, 4] = [0, -.0184, 0, 0, .35, .1, -.1, -.605, -.605, -.605]
    E[:, 5] = np.deg2rad([0, 0, r1, r2, 0, 0, 0, 0, 0, 0])
    return E
   



# def ct_shepp_logan_params_2d():
#     '''Return parameters for original Shepp-Logan phantom.

#     Returns
#     -------
#     E : array_like, shape (10, 6)
#         Parameters for the 10 ellipses used to construct the phantom.
#     '''

#     E = np.zeros((10, 6)) # (10, [A, a, b, xc, yc, theta])
#     E[:, 0] = [.01, .01, -.02, -.02, .01, .01, .01, .01, .01, .01]
#     # E[:, 1] = [
#     #     .69, .6624, .11, .16, .21, .046, .046, .046, .023, .023]
#     E[:, 1] = [
#         .58, .54, .11, .16, .16, .046, .046, .046, .023, .023]
#     E[:, 2] = [.92, .874, .31, .41, .25, .046, .046, .023, .023, .046]
#     E[:, 3] = [0, 0, .22, -.22, 0, 0, 0, -.08, 0, .06]
#     E[:, 4] = [0, -.0184, 0, 0, .35, .1, -.1, -.605, -.605, -.605]
#     E[:, 5] = np.deg2rad([0, 0, 18, -18, 0, 0, 0, 0, 0, 0])
#     return E



def ct_modified_shepp_logan_params_2d(r1, r2, ma1, mi1, ma2, mi2):
    '''Return parameters for modified Shepp-Logan phantom.

    Returns
    -------
    E : array_like, shape (10, 6)
        Parameters for the 10 ellipses used to construct the phantom.
    '''
    E = ct_shepp_logan_params_2d(r1, r2, ma1, mi1, ma2, mi2)
    E[:, 0] = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    return E


r2_range = np.linspace(0, 180, 36, False)
r2_label = np.random.randint(0, 36, 3000)

#r1_range =  np.linspace(0, 180,3000,False)
#np.random.shuffle(r1_range)

#r2_range = np.linspace(-180, 0,3000,False)
#np.random.shuffle(r2_range)

# ma1_range = np.linspace(.06, .10, 3000,False)
# np.random.shuffle(ma2_range)

# mi1_range = np.linspace(.24, .28, 3000,False)
# np.random.shuffle(mi2_range)

ma0_range = np.random.randint(110, 128, 3000)
np.random.shuffle(ma0_range)


ma2_range = np.linspace(.05, .08, 3000,False)
np.random.shuffle(ma2_range)

mi2_range = np.linspace(.15, .20, 3000,False)
np.random.shuffle(mi2_range)


rotation_range = np.linspace(0,180,36,False)
rotation_lable = np.random.randint(0, 36, 3000)
# scale_range = np.random.randint(110, 128, 3000)

P_all = []
L = []
ROI = []
EQUAL = []

D_all = []




for i in range(3000):
    ph = ct_shepp_logan_2d(ma0_range[i], ma0_range[i], 0, r2_range[r2_label[i]], .10, .28, ma2_range[i], mi2_range[i])
    
    # print(r1_range[r1_label[i]])
    N, M, theta1, theta2, theta3 = ma0_range[i], ma0_range[i], 0, np.deg2rad(0), np.deg2rad([r2_range[r2_label[i]]])
    xc1, yc1, xc2, yc2, xc3, yc3 = 0, 0, .22, 0, -.22, 0


    X, Y = np.meshgrid( # meshgrid needs linspace in opposite order
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, M))

    roi1 = np.zeros((ma0_range[i], ma0_range[i]))
    roi2 = np.zeros((ma0_range[i], ma0_range[i]))
    roi3 = np.zeros((ma0_range[i], ma0_range[i]))
    
    ct1 = np.cos(theta1)
    st1 = np.sin(theta1)
    
    ct2 = np.cos(theta2)
    st2 = np.sin(theta2)
    
    ct3 = np.cos(theta3)
    st3 = np.sin(theta3)
    
    ma_r1, mi_r1, ma_r2, mi_r2, ma_r3, mi_r3  = .52, .82, .10, .28, ma2_range[i], mi2_range[i]

    idx1 = (((X - xc1)*ct1 + (Y - yc1)*st1)**2/ma_r1**2 +((X - xc1)*st1 - (Y - yc1)*ct1)**2/mi_r1**2 < 1) 

# idx0_1 = (Y <= -54./128)
# idx0_2 = (Y >= 54./128)
    idx2 = (((X - xc2)*ct2 + (Y - yc2)*st2)**2/ma_r2**2 +((X - xc2)*st2 - (Y - yc2)*ct2)**2/mi_r2**2 < 1) 
    idx3 = (((X - xc3)*ct3 + (Y - yc3)*st3)**2/ma_r3**2 +((X - xc3)*st3 - (Y - yc3)*ct3)**2/mi_r3**2 < 1) 
    
    
    # rotation_e = rotation_range[rotation_lable[i]] - 18
    # idx2 = ndimage.rotate(idx2, rotation_e, reshape=False)
    
    roi1[idx1] = 1
    roi2[idx2] = 1
    roi3[idx3] = 1   
    
    
    roi1 = padding(roi1, 128, 128)
    roi2 = padding(roi2, 128, 128)
    roi3 = padding(roi3, 128, 128)
    ph = padding(ph, 128, 128)
        
    
    
    
    ph[roi1==0] = 0
    ph[roi2==1] = 0.2
    ph[roi3==1] = 0
    # ph[ph<0] = 0
    # ph = padding(ph, 128, 128)
    # label_n = rotation_range[rotation_lable[i]]
     
    label_n = rotation_range[rotation_lable[i]]
     
    ph = ndimage.rotate(ph, label_n, reshape=False)
    ph[ph<0.01] = 0
    
    
    P_all.append(ph)
    D_all.append(roi3)
   

np.save('Shepp_data_Internal_defect.npy', P_all)
np.save('Defect_Internal.npy', D_all)


