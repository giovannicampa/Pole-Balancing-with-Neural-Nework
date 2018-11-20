import gym
import pygame
import pandas as pd
import numpy as np
import random
env = gym.make("CartPole-v0")
training_steps = 300 #length of each training instance = timesteps 
goal_steps = 50 #timesteps that I want to be able to balance: training instances that can balance the pole for t> goal_steps, are used for training of the NN
trials_nr = 10000 #training instances: how many times the game is played randomly
scores = []
accepted_scores = [] #to know how well my training data performed
discount_rate = 0.99 #will influence how the total future reward of an action in influenced
one_step_reward = 1 #what you get to stay up for one timestep
training_data = pd.DataFrame()
env.reset()
reward_sum = [] #list that contains the sum of the discounted future rewards for each action; Is renewed at each episode

for i_episode in range(trials_nr): #iterate through all the episodes
    
    for t in range(training_steps): #each episode lasts max this nr of training (time) steps
        env.render()
        action = random.randrange(0,2) #takes a random action in the environment = (0,1)
        observation, reward, done, info = env.step(action) #array representing final state, 0-1, if game is over
        observation = observation.tolist() #observation is a numpy array, but to deal with other data, now it is converted to list
    env.reset()
