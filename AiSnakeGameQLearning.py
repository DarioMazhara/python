from ctypes.wintypes import POINT
from importlib.util import module_for_loader
from math import gamma
import os
from pickletools import optimize
from pyexpat import model
from tarfile import BLOCKSIZE
from turtle import forward
import turtle
import pygame
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

"""Part1
Initializing the linear neural network.
2. The function forward is used to take the input(11 state vector) and pass it through the 
   Neural network and apply relu activation function and give the output back i.e the next 
   move of 1 x 3 vector size. In short, this is the prediction function that would be called by the agent.
3. The save function is used to save the trained model for future use"""
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(save, file_name='model_name.pth'):
        model_folder_path = 'Path'
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)        
"""Part 2
1. Initialising QTrainer class
   ∗ setting the learning rate for the optimizer.
   * Gamma value that is the discount rate used in Bellman equation.
   * initialising the Adam optimizer for updation of weight and biases.
   * criterion is the Mean squared loss function.
2. Train_step function 
   * As you know that PyTorch work only on tensors, so we are converting all the input
    to tensors.
   * As discussed above we had a short memory training then we would only pass one value
    of state, action, reward, move so we need to convert them into a vector, so we had used
    unsqueezed function .
   * Get the state from the model and calculate the new Q value using the below formula:
                   Q_new = reward + gamma * max(next_predicted Qvalue)
   * calculate the mean squared error between the new Q value and previous Q value and 
   backpropagate that loss for weight updation."""
   
class QTrainer:
    def __init__(self.model, lr, gamma):
        # Learning rate for optimizer
        self.lr
        # Discount rate
        self.gamma = gamma
        # Linear Neural Network from above
        self.model = model
        # Optimizer to update weight and biases
        self.optimizer = nn.optim.Adam(model.parameters(), lr = self.lr)
        # Mean squared error loss function
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
            state = torch.tensor(state, dtype = torch.float)
            next_state = torch.tensor(next_state, dtype = torch.float)
            action = torch.tensor(action, dtype = torch.long)
            reward = torch.tensor(reward, dtype = torch.float)
            # Convert to tuple of shape (1, x) because only one param to train
            if(len(state.shape) == 1):
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.squeeze(reward, 0)
                done = (done, )
    
    # 1. Predicted Q val with curr. state
            pred = self.model(state)
            target = pred.clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.mode(next_state[idx]))
                target[idx][torch.argmax(action).item()] = Q_new
    # 2. Q_new = reward + gamma * max(next_predicted Qvalue)
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()
            self.optimizer.step()
            
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d),
             
            # Danger left
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)
        
    def get_action(self, state):
        # random moves
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if (random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).cuda()
            prediction = self.model(state0).cuda()
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, done = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
self.model.load_state_dict(torch.load('PATH'))

             

























































"""
Algorithm:
We have snake and food on the board randomly placed.
Calculate the state of the snake using the 11 values. if any the condition is true then set that value to zero else set one.
!!How 11 states are defined
Based on the current Head position agent will calculate the 11 state values as described above.
After getting these state, agent would pass this to the model and get the next move to perform.
After executing the next state calculate the reward. Rewards are defined as below:
Eat food : +10
Game Over : -10
Else : 0
Update the Q value (will be discussed later) and Train the Model.
After analyzing the algorithm now we have to build the idea to proceed for coding this algorithm.
"""