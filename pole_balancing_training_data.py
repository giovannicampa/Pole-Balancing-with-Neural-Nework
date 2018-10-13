import gym
import pandas as pd
import random

env = gym.make("CartPole-v0")
training_steps = 500 #length of each training instance = timesteps 
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
    score = 0
    prev_obs = []
    episode_memory = pd.DataFrame()
    obs_list = []
    
    for t in range(training_steps): #each episode lasts max this nr of training (time) steps
        #env.render()
        prev_obs_act_list = []
        action = random.randrange(0,2) #takes a random action in the environment = (0,1)
        observation, reward, done, info = env.step(action) #array representing final state, 0-1, if game is over
        observation = observation.tolist() #observation is a numpy array, but to deal with other data, now it is converted to list
        score = score + reward
                
        if t > 0:
            for i in range(4):
                prev_obs_act_list.append(prev_obs[i])
            prev_obs_act_list.append(action)
            episode_memory = episode_memory.append([prev_obs_act_list])
            
        prev_obs = observation
        
        if done:
            break
        
   for i_timestep in range(len(episode_memory)): #[0,1,2...11]
        discounted_reward = 0
        for x in reversed(range(len(episode_memory)-i_timestep)): #[11,10,..,episode_memory - i_timestep]
            discounted_reward = discounted_reward + one_step_reward*discount_rate**x
        reward_sum.append(discounted_reward)
        
    training_data = training_data.append(episode_memory)
    env.reset()

training_data.columns = ["Obs1", "Obs2", "Obs3", "Obs4", "Act"]
training_data.reset_index(inplace = True, drop = True)
training_data_w_scores = pd.concat([training_data, reward_sum], axis = 1)

X = training_data_w_scores.loc[:, ["Obs1","Obs2", "Obs3", "Obs4", "Tot_Rew"]]
y = training_data_w_scores.loc[:,["Act"]]


X.to_csv("X")
y.to_csv("y")


