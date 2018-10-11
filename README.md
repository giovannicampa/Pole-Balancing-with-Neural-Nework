# Pole-Balancing-with-Neural-Nework
Solving simple control problem with Neural Networks (modules used: gym, sklearn)

- pole_balancing_training_data.py
This file creates the data used to then train the NN. This is done by moving the pole according to random actions. In the end the good actions (score > 50) are filtered out and saved in two files X (previous observation) and y (action, that, according to the previous observation had the effet to keep the pole balanced).

- pole_balancing_training.py
Here the data from X and y is loaded and used to train a NN. Its parameters have been chosen by looking at the maximal performance reached by the algorithm in balancing the pole.
