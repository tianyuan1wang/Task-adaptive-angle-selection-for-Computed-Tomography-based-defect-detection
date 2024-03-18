from skimage.draw import polygon
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import argparse

import skimage
from skimage import draw
from scipy.ndimage import zoom
import numpy as np

import matplotlib.pyplot as plt

import random

import os

import torch

import astra

import math

import cv2

import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from copy import deepcopy

from skimage.metrics import structural_similarity as ssim

def seed(seed=1029):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



seed()



def load_raw_data(file_path, shape):
    """
    Load data from a binary .raw file.

    Parameters:
    - file_path: Path to the .raw file.
    - shape: Shape of the data (dimensions).
    - dtype: Data type of the elements.

    Returns:
    - Loaded data as a NumPy array.
    """
    try:
        # Read binary data
        data = np.fromfile(file_path,dtype=np.uint16)
        
        # Reshape the data according to the specified shape
        data = data.reshape(shape)
        
        return data

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

tmp0 = load_raw_data(r"Head3.raw",(128,128))
# Normalize the array
tmp0_min = tmp0.min()
tmp0_max = tmp0.max()
normalized_tmp0 = (tmp0 - tmp0_min) / (tmp0_max - tmp0_min)    
    
    
def midpoint_displacement(x1, y1, x2, y2, variance):
    # Calculate midpoint and offset its y-coordinate
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2 + np.random.uniform(-variance, variance)

    return mid_x, mid_y

def generate_crack(start, end, variance, iterations):
    points = [start, end]
    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            mid_x, mid_y = midpoint_displacement(x1, y1, x2, y2, variance)
            new_points.extend([(x1, y1), (mid_x, mid_y)])
        new_points.append(end)
        points = new_points

    return points

def draw_crack(crack_points, size):
    crack_image = np.zeros((size, size))

    for i in range(len(crack_points) - 1):
        x1, y1 = crack_points[i]
        x2, y2 = crack_points[i + 1]
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        crack_image = cv2.line(crack_image, (x1, y1), (x2, y2), 1, thickness=1)

    return crack_image

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
    

def dice_score(pred_mask, gt_mask):
    # Flatten the masks to 1D arrays
    pred_flat = pred_mask.ravel()
    gt_flat = gt_mask.ravel()

    # Calculate intersection and sums of the masks
    intersection = np.sum(pred_flat * gt_flat)
    mask_sum = np.sum(pred_flat) + np.sum(gt_flat)

    # Calculate Dice score
    dice = (2.0 * intersection) / (mask_sum + 1e-8)  # Adding a small value to avoid division by zero

    return dice
   
    
    
    

P_all = []

crack_image = []

roi_image = []

size = 128
start_point = (75, 40)
end_point = (60, 30)
variance = 1
iterations = 5


# The ranges of rotations and scales
scale_range = np.random.randint(110, 129, 900)

rotation_range = np.linspace(0,180,36,False)
rotation_label = np.random.randint(0, 36, 900)

# Loop through the files and read each one
# Loop through the files and read each one
for i in range(0,900):
    # Generate crack points using midpoint displacement
    crack_points = generate_crack(start_point, end_point, variance, iterations)
    # Draw the crack-like structure
    crack_data = draw_crack(crack_points, size)
    normalized_crack = deepcopy(normalized_tmp0)
    normalized_crack[crack_data.astype(bool)] = 0 
    crack_data = normalized_tmp0 - normalized_crack
        
    roi = np.zeros_like(normalized_crack, dtype=bool)
    roi[20:50, 50:75] = True
    crack_roi_data = deepcopy(normalized_crack)
    crack_roi_data[~roi] = 0    
    # Calculate the zoom factors for each dimension
    zoom_factor = scale_range[i] / 128

    # Resize the array
    normalized_crack = zoom(normalized_crack, zoom_factor)
    crack_roi_data = zoom(crack_roi_data, zoom_factor)
    crack_data = zoom(crack_data, zoom_factor)
        
    # Make sure size 128X128
    normalized_crack = padding(normalized_crack, 128, 128)
    crack_roi_data = padding(crack_roi_data, 128, 128)
    crack_data = padding(crack_data, 128, 128)

    # Rotate the array
    label_n = rotation_range[rotation_label[i]]
    normalized_crack = ndimage.rotate(normalized_crack, label_n, reshape=False)
    crack_roi_data = ndimage.rotate(crack_roi_data, label_n, reshape=False)
    crack_data = ndimage.rotate(crack_data, label_n, reshape=False)
    
    normalized_crack[normalized_crack<0.5] = 0
    crack_roi_data[crack_roi_data<0.5] = 0
    crack_data[crack_data<0.5] = 0
    crack_data[crack_data>=0.5] = 1

    P_all.append(normalized_crack)
    crack_image.append(crack_data) 
    roi_image.append(crack_roi_data)

    
# Prepare an empty list to store the numpy arrays
#nondefect_image = []
#nondefect_roi_image = []    
    
# Loop through the files and read each one
for i in range(0,900):
    nondefect_data = deepcopy(normalized_tmp0)
    roi = np.zeros_like(nondefect_data, dtype=bool)
    roi[20:50, 50:75] = True
    nondefect_roi_data = deepcopy(nondefect_data)
    nondefect_roi_data[~roi] = 0    
    # Calculate the zoom factors for each dimension
    zoom_factor = scale_range[i] / 128

    # Resize the array
    nondefect_data = zoom(nondefect_data, zoom_factor)
    nondefect_roi_data = zoom(nondefect_roi_data, zoom_factor)
    #b_pore_data = zoom(b_pore_data, zoom_factor)
        
    # Make sure size 128X128
    nondefect_data = padding(nondefect_data, 128, 128)
    nondefect_roi_data = padding(nondefect_roi_data, 128, 128)
    #b_pore_data = padding(b_pore_data, 128, 128)
        
    # Rotate the array
    label_n = rotation_range[rotation_label[i]]
    nondefect_data = ndimage.rotate(nondefect_data, label_n, reshape=False)
    nondefect_roi_data = ndimage.rotate(nondefect_roi_data, label_n, reshape=False)
    #b_pore_data = ndimage.rotate(b_pore_data, label_n, reshape=False)
    nondefect_data[nondefect_data<0.5] = 0
    nondefect_roi_data[nondefect_roi_data<0.5] = 0

    P_all.append(nondefect_data)
    roi_image.append(nondefect_roi_data)
    #b_pore_image.append(b_pore_data)  






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

        self.reward, self.ssim_reward, self.dic = self._get_reward(self.angles_seq)





        


        if self.a_start >= 20 or self.ssim_reward > 0.45:
            self.done = True
            self.a_start = 0
            self.angles_seq = []

       

        #self.previous_reward = self.current_reward

        return np.array(self.state), self.reward, self.done, self.angle_action, self.n, self.ssim_reward, self.dic


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
            self.state_reshaped = self.state.reshape(-1,1)
            # Find the index of the maximum value
            self.max_index = np.argmax(self.state_reshaped)
            self.max_index_2d = np.unravel_index(self.max_index, self.state_reshaped.shape)

            # Use the index as the initial center for KMeans
            self.initial_center = self.state_reshaped[self.max_index_2d]

            # Flatten the array for KMeans clustering
            self.flattened_array = self.state_reshaped.flatten().reshape(-1, 1)

            # Reshape the initial center to make it a 2D array
            self.initial_center = self.initial_center.reshape(1, -1)

            # KMeans clustering with the initial center
            self.kmeans = KMeans(n_clusters=2, n_init=1, random_state=0,             init=np.vstack((self.initial_center,                                 self.initial_center))).fit(self.flattened_array)
            self.labels = self.kmeans.labels_.reshape(self.state.shape)

            self.crack_bool = np.asarray(crack_image[self.n], dtype=bool)

            # Invert the boolean mask
            self.inverted_mask = ~self.crack_bool

            # Update labels using the inverted mask
            self.label_roi = deepcopy(self.labels)
            self.label_roi[self.inverted_mask] = 0
            #label_roi[b_pore_bool] = 0
            self.dic = dice_score(self.label_roi, crack_image[self.n])
        else:
            self.dic = 0
          
        #self.weight1 = self.a_start / 20.0
        #self.weight2 = 1-self.a_start / 20.0

        self.ssim_reward = ssim(P_all[self.n][roi_image[self.n].astype(bool)], self.state[roi_image[self.n].astype(bool)], data_range=np.max(P_all[self.n][roi_image[self.n].astype(bool)]))
        
       


        self.previous_action=self.angles_seq[-1]
        
        if self.a_start >= 20 or self.ssim_reward > 0.45:
            self.reward = (self.dic + self.ssim_reward) * 15
        else:
            self.reward = -1
       



        return self.reward, self.ssim_reward, self.dic



