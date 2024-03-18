import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim 

import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import random
import pickle
import numpy as np



seed = 1029


import os, sys


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("Environment_seperate_new_more_rotation_{}_01.py".format(seed)))))

from Pore_environment import *


NUM_EPISODES = 300000
LEARNING_RATE = 0.001


# env = gym.make('CartPole-v0')
env = env()
# nA = env.action_space.n




total_rewards = []
A = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

P_A = []
NUM = []



GAMMA = 0.99



episode_rewards = []
A = []
P_A = []
P_S = []

NUM = []
E = []
P_N = []
STATE = []
NUM_A = []
note = 'Head_defect'
VALUES = []
#VALUES2 = []
N_VALUES = []
#N_VALUES2 = []
DIC = []


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_1, output_dim, n_layers=2):
        super(ActorCritic, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
       
        to_pad = int((3 - 1) / 2)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, 
                          out_channels=12, 
                          kernel_size=3,
                          padding=to_pad,
                          stride=2),
                nn.GroupNorm(num_channels=12, 
                              num_groups=4),
                # nn.BatchNorm2d(12),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=12, 
                          out_channels=24, 
                          kernel_size=3,
                          stride=1, 
                          padding=to_pad),
                nn.GroupNorm(num_channels=24, num_groups=4),
                # nn.BatchNorm2d(24),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=24, 
                          out_channels=48, 
                          kernel_size=3,
                          stride=1, 
                          padding=to_pad),
                nn.GroupNorm(num_channels=48, num_groups=4),
                # nn.BatchNorm2d(24),
                nn.LeakyReLU(0.2),
               
                nn.MaxPool2d(kernel_size=4))             
        
        
        
        self.actor = nn.Sequential(
                                  nn.Linear(768+hidden_dim_1, output_dim),
                                  nn.Softmax(dim=-1))
        
        
        
        self.critic = nn.Sequential(
                                    nn.Linear(768+hidden_dim_1, 768+hidden_dim_1),
                                    nn.ReLU(),
                                    nn.Linear(768+hidden_dim_1, 1))
        

        
        
        
        
    def forward(self,state, state_a):
        state = torch.from_numpy(state).float().squeeze().to(device)
        #state1 = state.reshape(-1, image_size, image_size)
        state = state.unsqueeze(0)
        state = state.unsqueeze(0)
        state_a = torch.from_numpy(state_a).float().to(device)
        #h0 = torch.zeros(self.n_layers, state1.size(0),self.hidden_dim).to(device)
        #print(state.shape)
        p2 = self.conv1(state)
        
        
        p2 = p2.view(p2.size(0), -1)
        # print(p2.shape)
        p3 = torch.cat((p2, state_a),1)
        value = self.critic(p3)
        
        probs = self.actor(p3)
        dist = Categorical(probs)
        return dist, value

actor_layer = 1
critic_layer = 2    
    
    
INPUT_DIM = image_size
HIDDEN_DIM = 4*INPUT_DIM + 1

OUTPUT_DIM = N_a
HIDDEN_DIM_1 = N_a

model = ActorCritic(INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM_1, OUTPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)



for e in range(NUM_EPISODES):
    state = env.reset()
    #state = state.flatten()[None,:]
    state_a = np.array([[0]*N_a])
    
    values = []
    #values2 = []
    n_values = []
    #n_values2 = []
    dic = []
    
    
    score = 0
    a_episode = []
    log_probs = []
    values = []
    rewards = []
    masks = []
    returns = []
    angle_prob = []
    Num_P = []
    e_change = []
    S = []
    p_n = []
    p_s = []
    num_angles = 0
   
    # entropy = 0
  
    while True:
        
        dist, value = model(state, state_a)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        angle_dist = dist.probs.detach().cpu().numpy()
        angle_prob.append(angle_dist)
        
        next_state, reward, done, _, n, c_r, c = env.step(action.item())
        num_angles += 1
        
        state_a[0][action] = 1
        #next_state = np.concatenate((next_state, state_a), axis=1)
     
        next_dist, next_value = model(next_state, state_a)
       
        
        advantage = reward + (1-done)*GAMMA*next_value - value
        #advantage2 = reward + (1-done)*GAMMA*next_value2 - value2
        
        #advantage = (last_c)*advantage1 + (1-last_c_r)*advantage2
        
        actor_loss = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2).mean()
       



        loss = actor_loss + 0.5*critic_loss - 0.01 * entropy 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        
        
        a_episode.append(action.item())
        Num_P.append(n)
        # p_s.append(rec_with_init[env.n][1])
        values.append(value.detach().item())
        #values2.append(value2.detach().item())
        n_values.append(next_value.detach().item())
        #n_values2.append(next_value2.detach().item())
        dic.append(c)
        #l_contrast.append(last_c.detach().item())
        
        
        e_change.append(entropy.item())
        
        score += reward
        #print(reward)
        state = next_state
        
        if NUM_EPISODES - e <= 200:
            S.append(state)
            p_n.append(env.n)

        #print('r_b', reward)
        if done:
            break
            
        
    A.append(a_episode)    
    P_A.append(angle_prob)
    discounted_reward = 0
    NUM.append(Num_P)
    
    VALUES.append(values)
    #VALUES2.append(values2)
    N_VALUES.append(n_values)
    #N_VALUES2.append(n_values2)
    DIC.append(dic)
    #L_CONTRAST.append(l_contrast)
    
    
    E.append(e_change)
    
    if NUM_EPISODES - e < 200:
        STATE.append(S)
    P_N.append(p_n)
    # P_S.append(p_s)
    NUM_A.append(num_angles)
   
 
    if e % 100 == 0:
        print("episode", e, 'num_a', num_angles, 'env n', env.n, 'c_s', env.ssim_reward, 'dic', env.dic, 'a_s', env.a_start)
        print("score", score, "entropy", entropy.item())
        # print("PSNR truth", rec_with_init[env.n][1], "phantom", env.n)
    
    if e % 10000 == 0:
        torch.save(model.state_dict(), 'actor_critic_{}_{}'.format(e, note))
        np.save("episode_rewards_A2C_{}_{}.npy".format(e, note),episode_rewards)
        #np.save("P_A_A2C_{}_{}.npy".format(e, note),P_A)
        with open("Num_P_A2C_{}_{}.pkl".format(e, note), 'wb') as file:
            pickle.dump(NUM, file)
        #np.save("Num_P_A2C_{}_{}.npy".format(e, note), NUM)
        with open("Actions_A2C_{}_{}.pkl".format(e, note), 'wb') as file:
            pickle.dump(A, file)        
        #np.save("Actions_A2C_{}_{}.npy".format(e, note),A)  
        #np.save("States_A2C_{}_{}.npy".format(e, note),STATE)      
        np.save("PN_A2C_{}_{}.npy".format(e, note),P_N)  
# np.save("PS_A2C_CNN_scale_more_rotation_0.01_{}actor{}criticR_65000_s{}_b{}_FPSNR.npy".format(actor_layer, critic_layer, seed, beta),P_S) 

        with open("Actions_A2C_{}_{}.pkl".format(e, note), 'wb') as file:
            pickle.dump(A, file)   
        with open("Values_A2C_{}_{}.pkl".format(e, note), 'wb') as file:
            pickle.dump(VALUES, file)    
        #np.save("Values_A2C_{}_{}.npy".format(e, note), VALUES)
        #np.save('Values2_{}_s{}_{}.npy'.format(e, seed, note), VALUES2)
        with open("N_Values_A2C_{}_{}.pkl".format(e, note), 'wb') as file:
            pickle.dump(N_VALUES, file)
        #np.save("N_Values_A2C_{}_{}.npy".format(e, note), N_VALUES)
        #np.save('N_Values2_{}_s{}_{}.npy'.format(e, seed, note), N_VALUES2)
        with open("Dic_A2C_{}_{}.pkl".format(e, note), 'wb') as file:
            pickle.dump(DIC, file)
        #np.save("Contrast_A2C_{}_{}.npy".format(e, note), CONTRAST)
        #np.save('L_Contrast_{}_s{}_{}.npy'.format(e, seed, note), L_CONTRAST)

        np.save("Num_A_A2C_{}_{}.npy".format(e, note), NUM_A)    
    
        
    episode_rewards.append(score)
  
    
np.save("episode_rewards_A2C_{}_{}.npy".format(NUM_EPISODES, note),episode_rewards)
with open("Num_P_A2C_{}_{}.pkl".format(NUM_EPISODES, note), 'wb') as file:
    pickle.dump(NUM, file)
#np.save("Num_P_A2C_{}_{}.npy".format(NUM_EPISODES, note), NUM) 
with open("Actions_A2C_{}_{}.pkl".format(NUM_EPISODES, note), 'wb') as file:
    pickle.dump(A, file)
#np.save("Actions_A2C_{}_{}.npy".format(NUM_EPISODES, note),A)  
np.save("States_A2C_{}_{}.npy".format(NUM_EPISODES, note),STATE)  
np.save("PN_A2C_{}_{}.npy".format(NUM_EPISODES, note),P_N)  
# np.save("PS_A2C_CNN_scale_more_rotation_0.01_{}actor{}criticR_65000_s{}_b{}_FPSNR.npy".format(actor_layer, critic_layer, seed, beta),P_S) 
with open("Values_A2C_{}_{}.pkl".format(NUM_EPISODES, note), 'wb') as file:
    pickle.dump(VALUES, file)
#np.save("Values_A2C_{}_{}.npy".format(NUM_EPISODES, note), VALUES)
#np.save('Values2_{}_s{}_{}.npy'.format(NUM_EPISODES, seed, note), VALUES2)
with open("N_Values_A2C_{}_{}.pkl".format(NUM_EPISODES, note), 'wb') as file:
    pickle.dump(N_VALUES, file)
#np.save("N_Values_A2C_{}_{}.npy".format(NUM_EPISODES, note), N_VALUES)
#np.save('N_Values2_{}_s{}_{}.npy'.format(NUM_EPISODES, seed, note), N_VALUES2)
with open("Dic_A2C_{}_{}.pkl".format(NUM_EPISODES, note), 'wb') as file:
    pickle.dump(DIC, file)
#np.save("Contrast_A2C_{}_{}.npy".format(NUM_EPISODES, note), CONTRAST)
#np.save('L_Contrast_{}_s{}_{}.npy'.format(NUM_EPISODES, seed, note), L_CONTRAST)

with open("Num_A_A2C_{}_{}.pkl".format(NUM_EPISODES, note), 'wb') as file:
    pickle.dump(NUM_A, file)


#np.save("Num_A_A2C_{}_{}.npy".format(NUM_EPISODES, note), NUM_A)    
with open("P_A_A2C_{}_{}.pkl".format(NUM_EPISODES, note), 'wb') as file:
    pickle.dump(P_A, file)
#np.save("P_A_A2C_{}_{}.npy".format(NUM_EPISODES, note),P_A)


