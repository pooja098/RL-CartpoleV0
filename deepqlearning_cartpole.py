#to install gym library- pip install gym
#to install tensorflow library- pip install tensorflow

import gym
import random
import numpy as np
import tflearn
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 0.00001 #Learning rate
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 200
score_requirement = 50
initial_games = 12000

#training data for neural network
def initial_population():

    training_data = []

    scores = []
    accepted_scores = []
    #play initial number of games to generate the training set
    for _ in range(initial_games):
        score = 0
        #state action sequence of a game
        game_memory = []

        prev_observation = []
        #an episode
        for _ in range(goal_steps):
            #pick random action
            action = random.randrange(0,2)
            #observe the environment after the action
            observation, reward, done, info = env.step(action)

            #append previous observation and action pair in game memory if not the first action of the episode
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        #if game is upto the requirement, add it to training set
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:

                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]


                training_data.append([data[0], output])

        #start new game
        env.reset()
        #add this useful score to the array
        scores.append(score)


    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)


    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):
    #setting up the neural network
    network = input_data(shape=[None, input_size, 1], name='input')
    #designing each layer with 256 nodes, 512 nodes and so on
    #activation function used is relu

    network = fully_connected(network, 256, activation='relu')
    #0.8 probability of nodes being retained
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 600, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 700, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 400, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    #output layer, outputs the action 
    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []
avg=[0]
x=[0]
#play 5000 games to test the trained model
for i in range(5000):
    score = 0
    game_memory = []
    prev_obs = []
    #start a new game
    env.reset()
    for _ in range(goal_steps):
        env.render()
        #first action of the game is random
        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            #predict the action using model
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

        choices.append(action)
        #observe the environment after the action
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)
    #average reward over last 100 episodes
    if i>=100:
      small=scores[i-100:i+1]
      ans=sum(small)/len(small)
      avg.append(ans)
      x.append(i-100)
      print(' game: {0} avg_last_100 : {1}'.format(i, ans))
      #if we achieve target of 195, stop playing
      if(ans>=195):
        print('Average Score: {}'.format(ans))
        break
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)
#plot avg reward of last 100 games for number of games played
plt.plot(x, avg)
plt.show()
