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
    
    
ma0_range = np.random.randint(110, 128, 900)
np.random.shuffle(ma0_range)


ma2_range = np.linspace(.05, .08, 900,False)
ma3_range = np.linspace(.08, .11, 900,False)
#np.random.shuffle(ma2_range)

mi2_range = np.linspace(.15, .20, 900,False)
mi3_range = np.linspace(.18, .23, 900,False)
#np.random.shuffle(mi2_range)


r2_range = np.linspace(0,180,36,False)
r2_label = np.random.randint(0, 36, 900)

rotation_range = np.linspace(0,180,36,False)
rotation_label = np.random.randint(0, 36, 900)
# scale_range = np.random.randint(110, 128, 3000)

P_all = []
L = []
D = []
B = []

ROI = []
EQUAL = []



for i in range(900):
    ph = ct_shepp_logan_2d(ma0_range[i], ma0_range[i], 0, r2_range[r2_label[i]], .10, .28, ma2_range[i], mi2_range[i])
    
    # print(r1_range[r1_label[i]])
    N, M, theta1, theta2, theta3, theta4 = ma0_range[i], ma0_range[i], 0, np.deg2rad(0), np.deg2rad([r2_range[r2_label[i]]]), np.deg2rad([r2_range[r2_label[i]]])
    xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4 = 0, 0, .22, 0, -.22, 0, -.22, 0


    X, Y = np.meshgrid( # meshgrid needs linspace in opposite order
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, M))

    roi1 = np.zeros((ma0_range[i], ma0_range[i]))
    roi2 = np.zeros((ma0_range[i], ma0_range[i]))
    roi3 = np.zeros((ma0_range[i], ma0_range[i]))
    roi4 = np.zeros((ma0_range[i], ma0_range[i]))
    
    ct1 = np.cos(theta1)
    st1 = np.sin(theta1)
    
    ct2 = np.cos(theta2)
    st2 = np.sin(theta2)
    
    ct3 = np.cos(theta3)
    st3 = np.sin(theta3)
    
    ct4 = np.cos(theta4)
    st4 = np.sin(theta4)    
    
    ma_r1, mi_r1, ma_r2, mi_r2, ma_r3, mi_r3, ma_r4, mi_r4  = .52, .82, .10, .28, ma2_range[i], mi2_range[i], ma3_range[i], mi3_range[i]

    idx1 = (((X - xc1)*ct1 + (Y - yc1)*st1)**2/ma_r1**2 +((X - xc1)*st1 - (Y - yc1)*ct1)**2/mi_r1**2 < 1) 

# idx0_1 = (Y <= -54./128)
# idx0_2 = (Y >= 54./128)
    idx2 = (((X - xc2)*ct2 + (Y - yc2)*st2)**2/ma_r2**2 +((X - xc2)*st2 - (Y - yc2)*ct2)**2/mi_r2**2 < 1) 
    idx3 = (((X - xc3)*ct3 + (Y - yc3)*st3)**2/ma_r3**2 +((X - xc3)*st3 - (Y - yc3)*ct3)**2/mi_r3**2 < 1) 
    idx4 = (((X - xc4)*ct4 + (Y - yc4)*st4)**2/ma_r4**2 +((X - xc4)*st4 - (Y - yc4)*ct4)**2/mi_r4**2 < 1)
    
    # rotation_e = rotation_range[rotation_lable[i]] - 18
    # idx2 = ndimage.rotate(idx2, rotation_e, reshape=False)
    
    roi1[idx1] = 1
    roi2[idx2] = 1
    roi3[idx3] = 1   
    roi4[idx4] = 1   
    
    roi1 = padding(roi1, 128, 128)
    roi2 = padding(roi2, 128, 128)
    roi3 = padding(roi3, 128, 128)
    roi4 = padding(roi4, 128, 128)
    ph = padding(ph, 128, 128)
        
    
    
    
    ph[roi1==0] = 0
    ph[roi2==1] = 0.2
    ph[roi3==1] = 0
    # ph[ph<0] = 0
    # ph = padding(ph, 128, 128)
    # label_n = rotation_range[rotation_lable[i]]
     
    label_n = rotation_range[rotation_label[i]]
     
    ph = ndimage.rotate(ph, label_n, reshape=False)
    ph[ph<0.01] = 0
    
    roi3 = ndimage.rotate(roi3, label_n, reshape=False)
    roi3[roi3<0.01] = 0 
    roi3[roi3>0.9] = 1  
    
    roi4 = ndimage.rotate(roi4, label_n, reshape=False)
    roi4[roi4<0.01] = 0 
    roi4[roi4>0.9] = 1  
    
    
    P_all.append(ph)
    D.append(roi3)
    B.append(roi4)
    
    
for i in range(900):
    ph = ct_shepp_logan_2d(ma0_range[i], ma0_range[i], 0, r2_range[r2_label[i]], .10, .28, ma2_range[i], mi2_range[i])
    
    # print(r1_range[r1_label[i]])
    N, M, theta1, theta2, theta3, theta4 = ma0_range[i], ma0_range[i], 0, np.deg2rad(0), np.deg2rad([r2_range[r2_label[i]]]), np.deg2rad([r2_range[r2_label[i]]])
    xc1, yc1, xc2, yc2, xc3, yc3, xc4, yc4 = 0, 0, .22, 0, -.22, 0, -.22, 0


    X, Y = np.meshgrid( # meshgrid needs linspace in opposite order
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, M))

    roi1 = np.zeros((ma0_range[i], ma0_range[i]))
    roi2 = np.zeros((ma0_range[i], ma0_range[i]))
    roi3 = np.zeros((ma0_range[i], ma0_range[i]))
    roi4 = np.zeros((ma0_range[i], ma0_range[i]))
    
    ct1 = np.cos(theta1)
    st1 = np.sin(theta1)
    
    ct2 = np.cos(theta2)
    st2 = np.sin(theta2)
    
    ct3 = np.cos(theta3)
    st3 = np.sin(theta3)
    
    ct4 = np.cos(theta4)
    st4 = np.sin(theta4)    
    
    ma_r1, mi_r1, ma_r2, mi_r2, ma_r3, mi_r3, ma_r4, mi_r4  = .52, .82, .10, .28, ma2_range[i], mi2_range[i], ma3_range[i], mi3_range[i]

    idx1 = (((X - xc1)*ct1 + (Y - yc1)*st1)**2/ma_r1**2 +((X - xc1)*st1 - (Y - yc1)*ct1)**2/mi_r1**2 < 1) 

# idx0_1 = (Y <= -54./128)
# idx0_2 = (Y >= 54./128)
    idx2 = (((X - xc2)*ct2 + (Y - yc2)*st2)**2/ma_r2**2 +((X - xc2)*st2 - (Y - yc2)*ct2)**2/mi_r2**2 < 1) 
    idx3 = (((X - xc3)*ct3 + (Y - yc3)*st3)**2/ma_r3**2 +((X - xc3)*st3 - (Y - yc3)*ct3)**2/mi_r3**2 < 1) 
    # idx4 = (((X - xc4)*ct4 + (Y - yc4)*st4)**2/ma_r4**2 +((X - xc4)*st4 - (Y - yc4)*ct4)**2/mi_r4**2 < 1)
    
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
    ph[roi3==1] = 0.2
    # ph[ph<0] = 0
    # ph = padding(ph, 128, 128)
    # label_n = rotation_range[rotation_lable[i]]
     
    label_n = rotation_range[rotation_label[i]]
     
    ph = ndimage.rotate(ph, label_n, reshape=False)
    ph[ph<0.01] = 0
    
    
    P_all.append(ph)
    # D.append(roi3)
    # B.append(roi4)    
    
    
def Gauss_noise(P, proj_angles, proj_size, vol_geom, percentage=0.05):
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, proj_angles)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    W = astra.OpTomo(proj_id)
    sinogram = W * P
    n = np.random.normal(0, sinogram.std(), (len(proj_angles), proj_size)) * percentage

    return n



def reconstruction_noise(P, proj_angles, proj_size, vol_geom, n_iter_sirt, distance_source_detector, distance_source_origin, percentage=0.0):

    proj_geom = astra.create_proj_geom('fanflat', 1.0, proj_size, proj_angles, distance_source_detector, distance_source_origin)

    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

    W = astra.OpTomo(proj_id)

    sinogram = W * P

    sinogram = sinogram.reshape([len(proj_angles), proj_size])



    n = np.random.normal(0, sinogram.std(), (len(proj_angles), proj_size)) * percentage


    sinogram_n = sinogram + n



    rec_sirt = W.reconstruct('SIRT_CUDA', sinogram_n, iterations=n_iter_sirt, extraOptions={'MinConstraint':0.0,'MaxConstraint':1.0})





    return rec_sirt



def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    #print(rmse)
    return 20*math.log10(1.0/rmse)

def angle_range(N_a):
    return np.linspace(0,2*np.pi,N_a,False)


a_start = 0



image_size = 128

proj_size = int(2*image_size)

vol_geom = astra.create_vol_geom(image_size, image_size)

n_iter_sirt = 150

distance_source_detector = 400

distance_source_origin = 200



N_a = 360

angles = angle_range(N_a)
# k = 0
# rec = reconstruction_noise(P_all[k], np.linspace(0,2*np.pi,360,False), proj_size, vol_geom, n_iter_sirt, distance_source_detector, distance_source_origin)
# contrast = (rec[(B[k]-D[k]).astype(bool)].mean() - rec[D[k].astype(bool)].mean())/rec[(B[k]-D[k]).astype(bool)].mean()   

#contrast = (rec[(B[k]).astype(bool)].mean() - rec[D[k].astype(bool)].mean())/rec[(B[k]).astype(bool)].mean()   

# contrast = np.abs(contrast)
# print(contrast)


# plt.figure()
# plt.imshow(P_all[k])

# plt.figure()
# plt.imshow(B[k])

# plt.figure()
# plt.imshow(B[k]-D[k])

# plt.figure()
# plt.imshow(rec)

class env():



    def __init__(self):

        self.n = np.random.randint(0,1800)

        self.criteria = 0
        
        self.total_reward = 0





    def step(self, action):



        # Wait for environment to transition to next state

        self.angle_action = angles[action]
        # Execute action on environment 

        self.a_start += 1



        self.angles_seq.append(self.angle_action)



        self.state = reconstruction_noise(P_all[self.n], self.angles_seq, proj_size, vol_geom, n_iter_sirt, distance_source_detector, distance_source_origin)




        # Get reward for new state

        self.reward, self.ssim_reward, self.contrast = self._get_reward(self.angles_seq)





        #self.total_reward += self.reward





        #self.criteria = psnr(P[self.n], self.state)







        # if  self.a_start > 6:

        #     # self.n = np.random.randint(0,4)

        #     self.a_start = 0

        #     self.angles_seq = []

        #     self.done = True


        if self.a_start >= 20 or self.ssim_reward > 0.85:
            self.done = True
            self.a_start = 0
            self.angles_seq = []

       

        #self.previous_reward = self.current_reward

        return np.array(self.state), self.reward, self.done, self.angle_action, self.n, self.ssim_reward, self.contrast


    def reset(self):

        self.n = np.random.randint(0,1800)



        self.a_start = a_start



        self.curr_iteration = 0



        self.total_reward = 0.0

        self.angles_seq = []



        #Default initialization for action

        self.previous_action=0

        self.previous_reward = 0
        
        self.total_reward = 0

        self.action=self.previous_action



        self.reward = 0



        self.done=False


        #self.n = 0





        self.state = np.zeros((128,128))

        self.criteria = 0

        return self.state




    def _get_reward(self,angles_seq):
       # reward = -0.01

        if self.n < 900:
          self.D_0 = D[self.n].astype(int)
          self.contrast = (self.state[B[self.n].astype(bool)].mean() -self.state[self.D_0.astype(bool)].mean())/self.state[B[self.n].astype(bool)].mean()
          self.contrast = np.abs(self.contrast)
        else:
          self.contrast = 0
          
        #self.weight1 = self.a_start / 20.0
        #self.weight2 = 1-self.a_start / 20.0

        self.ssim_reward = ssim(P_all[self.n], self.state, data_range=np.max(P_all[self.n]))
        
        #self.current_reward = self.weight1 * self.contrast + self.weight2 * self.ssim_reward
        
        #self.improve_reward = self.current_reward - self.previous_reward

        #self.total_reward += self.current_reward



        self.previous_action=self.angles_seq[-1]
        
        if self.a_start >= 20 or self.ssim_reward > 0.85:
            self.reward = (self.contrast + self.ssim_reward) * 15
        else:
            self.reward = -1
        #    if self.n < 900:
        #        reward = self.contrast
        #    else:
        #        reward = self.ssim_reward
            #reward = self.total_reward
        #else:
            
        #self.previous_reward = self.total_reward



        return self.reward, self.ssim_reward, self.contrast




