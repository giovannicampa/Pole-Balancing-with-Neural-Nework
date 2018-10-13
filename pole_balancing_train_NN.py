import gym
import random
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
#from sklearn.neighbors import KNeighborsRegressor this algorithm has also been tried, and it gives a bit worse performance with the same data
from sklearn.preprocessing import OneHotEncoder

X = pd.read_csv("X").loc[:, ["Obs1","Obs2", "Obs3", "Obs4", "Tot_Rew"]] #loading the training data
y = pd.read_csv("y_s").loc[:, ["Act"]]

X_good = X[X.Tot_Rew > min_score]
X_good_train = X_good.iloc[:,0:4]

encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y).toarray()
y_encoded_df = pd.DataFrame(y_encoded) #encoding the "action" data to make it compatible with the game
y_encoded_df.columns = ["ac1", "ac2"]
y_train = y_encoded_df[X.Tot_Rew > min_score]

reg = MLPRegressor(solver = "adam", activation = "relu", hidden_layer_sizes = 200) #hyperparameters have been choosen in a way to maximise the score 
#reg = KNeighborsRegressor(n_neighbors = 10)
reg.fit(X_good_train, y_train) #direct training with all the dataset
env = gym.make("CartPole-v0")

n_games = 10
goal_steps = 400
score = 0
score_list = []

for i in range(n_games):
    score = 0
    env.reset()
    prev_obs = []
    prev_obs_df = []
    for j in range(goal_steps):
        env.render()

        if j < 1: #starts with a random action because I have no information about the observation
            action = random.randrange(0,2)
            print("Random action:", action)
        else:
            action = np.argmax(reg.predict(prev_obs_df)) #saves the action predicted by the NN according to the previous observation
            print("Predicted action:", action)
        observation, reward, done, info = env.step(action)
        prev_obs = observation.tolist() #updates the previous observation
        prev_obs_df = pd.DataFrame([prev_obs])
        score += reward
        if done:
            score_list.append(score)
            break
print("The NN performed as follows:", score_list)
print("\nThe average performance of the NN is:", sum(score_list)/len(score_list))

