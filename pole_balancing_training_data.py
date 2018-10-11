import gym
import pandas as pd
import random

env = gym.make("CartPole-v0")
training_steps = 500 #length of each training instance = timesteps 
goal_steps = 50 #timesteps that I want to be able to balance: training instances that can balance the pole for t> goal_steps, are used for training of the NN
trials_nr = 10000 #training instances: how many times the game is played randomly
scores = []
accepted_scores = [] #to know how well my training data performed
training_data = pd.DataFrame()
env.reset()

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
        
    if score >= goal_steps: #here the good episodes are filtered out and used as training data
        accepted_scores.append(score)
        training_data = training_data.append(episode_memory)
    env.reset()
    
print(accepted_scores)
print(sum(accepted_scores)/len(accepted_scores))

training_data.columns = ["Obs1", "Obs2", "Obs3", "Obs4", "Act"]
training_data.reset_index(inplace = True, drop = True)

X = training_data.iloc[:,0:4]
y = training_data.iloc[:,4:5]


X.to_csv("X")
y.to_csv("y")


