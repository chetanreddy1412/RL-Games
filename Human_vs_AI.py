# Play Game

### Importing Libraries

import gym
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
from nqueens import Queen
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import random
import tqdm
import os
import gc
import copy

import cv2

from IPython.display import HTML
from collections import namedtuple, deque
from itertools import count
from base64 import b64encode
from pynput.keyboard import Listener

from IPython import display
import torch
from tqdm.notebook import tqdm
from torch import nn
from collections import deque,namedtuple
import warnings
warnings.filterwarnings("ignore")

### Environment

#in vasuki custom, the only change from "Vasuki" is the fact that history is looked one step behind while in the original environment, it is -2, check the "step" function

class Vasuki_custom(Env):

    def _food_position_(self, n):
        # Using the N-Queens problem to uniformly distribute the food spawning location
        qq = Queen(n)
        food_pos = np.empty(shape = [0, 2])
        chess = qq.queen_data[0]
        for x in range(n):
            for y in range(n):
                if chess[y][x] == 1:
                    arr = np.array([[x, y]])
                    food_pos = np.append(food_pos, arr, axis = 0)
        # Returning the n food locations which are spatially distributed uniformly
        return food_pos
    
    def _init_agent_(self, score=0):
        # Creating a dictionary to store the information related to the agent
        agent = {}
        # Set initial direction of head of the Snake :  North = 0, East = 1, South = 2, West = 3
        agent['head'] = np.random.randint(low = 0, high = 4, size = (1)).item()
        # The score for each agent
        agent['score'] = score
        # Set initial position 
        agent['state'] = np.random.randint(low = 0, high = self.n, size = (2))
        # Velocity of the snake
        agent['velocity'] = 1            
        # Returning the Agent Properties
        return agent

    def _init_image_(self, path):
        # Loading the image
        image = cv2.imread(path)
        # Resizing the image
        image = cv2.resize(image, (self.scale-1,self.scale-1), interpolation=cv2.INTER_NEAREST)
        # Returning the preprocessed image
        return image

    def __init__(self, n, rewards, game_length=100):
        # Parameters
        self.n = n
        self.rewards = rewards
        self.scale = 256//self.n
        # Actions we can take : left = 0, forward = 1, right = 2
        self.action_space = Discrete(3)
        # The nxn grid
        self.observation_space = MultiDiscrete([self.n, self.n])
        # Set Total Game length
        self.game_length = game_length
        self.game_length_ = self.game_length
        # Set Food Spawning locations. Totally there are only n locations
        self.foodspawn_space = self._food_position_(self.n)
        # Out of the n food locations, at any time only n/2 random locations have food
        self.live_index = np.random.choice(len(self.foodspawn_space), size=(self.n//2), replace=False)
        self.live_foodspawn_space = self.foodspawn_space[self.live_index]
        # Initializing the Agents
        self.agentA = self._init_agent_()
        self.agentB = self._init_agent_()
        # Loading the Images
        self.image_agentA = self._init_image_("agentA.png")
        self.image_agentB = self._init_image_("agentB.png")
        self.image_prey = self._init_image_("prey.png")
        # Creating History
        encoded, _ = self.encode()
        self.history = [] # {"agentA": self.agentA, "agentB":self.agentB, "live_foodspawn_space": self.live_foodspawn_space, 'encoded': encoded}

    def _movement_(self, action, agent):
        # Loading the states
        illegal = 0     # If the snake hits the walls
        n = self.n
        head = agent['head']
        state = agent['state'].copy()
        velocity = agent['velocity']
        score = agent['score']
        # Applying the Action
        if action == 0: # Go Left
            if head==0:
                if state[1]==velocity-1: # Left Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, -velocity])
                head = 3
            elif head==1:
                if state[0]==velocity-1: # Top Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([-velocity, 0])
                head = 0
            elif head==2: 
                if state[1]==n-velocity: # Right Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, velocity])
                head = 1
            elif head==3:
                if state[0]==n-velocity: # Bottom Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([velocity, 0])
                head = 2           
        elif action == 1: # Move Forward
            if head==0:
                if state[0]==velocity-1: # Top Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([-velocity, 0])
                head = 0
            elif head==1:
                if state[1]==n-velocity: # Right Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, velocity])
                head = 1
            elif head==2:
                if state[0]==n-velocity: # Bottom Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([velocity, 0])
                head = 2
            elif head==3:
                if state[1]==velocity-1: # Left Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, -velocity])
                head = 3
        elif action == 2: # Go Right
            if head==0:
                if state[1]==n-velocity: # Right Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, velocity])
                head = 1
            elif head==1:
                if state[0]==n-velocity: # Bottom Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([velocity, 0])
                head = 2
            elif head==2:
                if state[1]==velocity-1: # Left Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, -velocity])
                head = 3
            elif head==3:
                if state[0]==velocity-1: # Top Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([-velocity, 0])
                head = 0
        # Updating the agent properties
        modified = {'head': head, 'state':state+change, 'score':score, 'velocity':velocity}
        return modified, illegal

    def _reward_(self, agent, illegal):
        # Loading the states
        head = agent['head']
        state = agent['state'].copy()
        velocity = agent['velocity']
        score = agent['score']
        # Calculating the reward
        if illegal == 1: # If the snake hits the wall
            reward = self.rewards['Illegal']
        else:
            if True in np.all((state == self.live_foodspawn_space), axis = 1):
                # Finding the index of the state
                index = np.where(np.all((state == self.live_foodspawn_space), axis = 1) == True)[0].item()
                # Computing the empty foodspawn spaces
                empty_foodspawn_space = [space for space in self.foodspawn_space if space not in self.live_foodspawn_space]
                # Removing the state from live foodspawn space
                self.live_foodspawn_space = np.delete(self.live_foodspawn_space, index, 0)
                # Updating the live foodspawn space
                addition = np.random.choice(len(empty_foodspawn_space), size=1, replace=False)
                self.live_foodspawn_space = np.append(self.live_foodspawn_space, np.expand_dims(empty_foodspawn_space[addition.item(0)], axis = 0), axis=0)
                assert  len(set([(x,y) for (x,y) in self.live_foodspawn_space])) == 4
                # If the snake lands on the food
                reward = self.rewards['Food']
            else:
                # If the snake just moves
                reward = self.rewards['Movement']
        return reward

    def step(self, action):
        actionA = action['actionA']
        actionB = action['actionB']
        # Applying the actions
        self.agentA, illegalA = self._movement_(actionA, self.agentA)
        self.agentB, illegalB = self._movement_(actionB, self.agentB) 
        # Calculating the reward
        if (self.agentA['state'] == self.agentB['state']).all():
            if self.agentA['score'] > self.agentB['score']:
                rewardA = 5 * abs( self.agentB['score']//(self.agentA['score']-self.agentB['score']) )
                rewardB = - 3 * abs( self.agentB['score']//(self.agentA['score']-self.agentB['score']) )
                _ = self._reward_(self.agentA, illegalA)
                score = self.agentB['score']
                while True:
                    self.agentB = self._init_agent_(score)
                    if (self.agentB['state']!=self.agentA['state']).all():
                        _ = self._reward_(self.agentB, illegalB)
                        break
            elif self.agentA['score'] < self.agentB['score']:
                rewardA = - 3 * abs( self.agentA['score']//(self.agentA['score']-self.agentB['score']) )
                rewardB = 5 * abs( self.agentA['score']//(self.agentA['score']-self.agentB['score']) )
                _ = self._reward_(self.agentB, illegalB)
                score = self.agentA['score']
                while True:
                    self.agentA = self._init_agent_(score) 
                    if (self.agentA['state']!=self.agentB['state']).all():
                        _ = self._reward_(self.agentA, illegalA)
                        break
            elif self.agentA['score'] == self.agentB['score']:
                rewardA = - abs(self.agentA['score']//2)
                rewardB = - abs(self.agentB['score']//2)
                while True:
                    self.agentA = self._init_agent_(score=self.agentA['score'])
                    if (self.agentA['state']!=self.agentB['state']).all():
                        _ = self._reward_(self.agentA, illegalA)
                        break
                while True:
                    self.agentB = self._init_agent_(score=self.agentB['score'])
                    if (self.agentB['state']!=self.agentA['state']).all():
                        _ = self._reward_(self.agentB, illegalB)
                        break
        else:
            rewardA = self._reward_(self.agentA, illegalA)
            rewardB = self._reward_(self.agentB, illegalB)
        # Adding the reward to the score
        self.agentA['score'] = self.agentA['score'] + rewardA
        self.agentB['score'] = self.agentB['score'] + rewardB
        # Updating history
        encoded, _ = self.encode()
        self.history.append({"agentA": self.agentA, "agentB":self.agentB, "live_foodspawn_space": self.live_foodspawn_space, "encoded": encoded, 
                             "rewardA": rewardA, "actionA": actionA, "rewardB": rewardB, "actionB": actionB})
        # Check if game is done
        self.game_length -= 1
        if self.game_length <= 0:
            done = True
        else:
            done = False
        # Set placeholder for info
        info = {'agentA': self.agentA, 'agentB': self.agentB}
        return  rewardA, rewardB, done, info

    def _rotate_(self, image, direction):
        # Rotating the image to rectify the direction of the head
        if direction == 1:
            image = np.rot90(image.copy(), k = 3)
        elif direction == 2: 
            image = np.rot90(image.copy(), k = 2)
        elif direction == 3:
            image = np.rot90(image.copy())
        return image

    def render(self): # Returns a one-hot encoded state
        # Loading the states
        live_foodspawn_space_ = self.live_foodspawn_space
        agentA = self.agentA
        agentB = self.agentB
        snakeA = agentA['state']
        snakeB = agentB['state']
        # Initializing the state
        state = np.ones((self.scale*self.n, 2*self.scale*self.n, 3))*255
        # Adding grid lines
        for x in range(self.n+1):
            state[self.scale*x:self.scale*x+1, :self.scale*self.n] = [0, 0, 0]
        for y in range(self.n+1):
            state[:, self.scale*y:self.scale*y+1] = [0, 0, 0]
        # Adding the live food location
        assert  len(set([(x,y) for (x,y) in live_foodspawn_space_])) == 4
        for food in live_foodspawn_space_.tolist():
            x = int(food[0])
            y = int(food[1])
            state[self.scale*x+1:self.scale*x+self.scale, self.scale*y+1:self.scale*y+self.scale] = self.image_prey
        # Annotating
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        color = (0, 0, 0)
        thickness = 1
        direction = {0:"North", 1:"East", 2:"South", 3:"West"}
        action = {0:"Left", 1:"Forward", 2:"Right", "None":"None"}
        stateA = "State A: [{0},{1}]".format(snakeA[0], snakeA[1])
        stateB = "State B: [{0},{1}]".format(snakeB[0], snakeB[1])
        scoreA = "Score A: " + str(agentA['score'])
        scoreB = "Score B: " + str(agentB['score'])
        headA = "Head A: " + direction[agentA['head']]
        headB = "Head B: " + direction[agentB['head']]
        #actionA = "Action A: " + action[actionA]
        #actionB = "Action B: " + action[actionB]
        # Adding the text
        start = 80
        state = cv2.putText(state, scoreA, (265, start), font, fontScale, color, thickness, cv2.LINE_AA)
        state = cv2.putText(state, stateA, (265, start+32), font, fontScale, color, thickness, cv2.LINE_AA)
        state = cv2.putText(state, headA, (265, start+64), font, fontScale, color, thickness, cv2.LINE_AA)
        #state = cv2.putText(state, actionA, (265, start+96), font, fontScale, color, thickness, cv2.LINE_AA)
        state = cv2.putText(state, scoreB, (390, start), font, fontScale, color, thickness, cv2.LINE_AA)
        state = cv2.putText(state, stateB, (390, start+32), font, fontScale, color, thickness, cv2.LINE_AA)
        state = cv2.putText(state, headB, (390, start+64), font, fontScale, color, thickness, cv2.LINE_AA)
        #state = cv2.putText(state, actionB, (390, start+96), font, fontScale, color, thickness, cv2.LINE_AA)
        # Adding the agents
        image_agentA = self._rotate_(self.image_agentA, agentA['head'])
        image_agentB = self._rotate_(self.image_agentB, agentB['head'])
        state[self.scale*snakeA[0]+1:self.scale*snakeA[0]+self.scale, self.scale*snakeA[1]+1:self.scale*snakeA[1]+self.scale] = image_agentA
        state[self.scale*snakeB[0]+1:self.scale*snakeB[0]+self.scale, self.scale*snakeB[1]+1:self.scale*snakeB[1]+self.scale] = image_agentB
        # Returning the state
        return state

    def encode(self):
        # Loading the states
        encoder = {'blank': 0, 'foodspawn_space': 1, 'agentA': 2, 'agentB': 3}
        state = np.zeros((self.n, self.n))
        live_foodspawn_space = self.live_foodspawn_space.astype(np.int)
        snakeA = self.agentA['state']
        snakeB = self.agentB['state']
        # Adding the agents and snakes
        state[live_foodspawn_space[:,0], live_foodspawn_space[:,1]] = encoder['foodspawn_space']
        state[snakeA[0], snakeA[1]] = encoder['agentA']
        state[snakeB[0], snakeB[1]] = encoder['agentB']
        # One-Hot encoding the state
        encoded = np.eye(len(encoder.keys()))[state.astype(np.int)]
        encoded = np.moveaxis(encoded, -1, 0)
        # Returning the encoded and state
        return encoded, state

    def reset(self):
        # Reset Total Game length
        self.game_length = self.game_length_
        # Reset Food Spawning locations
        self.foodspawn_space = self._food_position_(self.n)
        # Reset Live Food Spawning locations
        self.live_index = np.random.choice(len(self.foodspawn_space), size=(self.n//2), replace=False)
        self.live_foodspawn_space = self.foodspawn_space[self.live_index]
        # Reset Agents
        self.agentA = self._init_agent_()
        self.agentB = self._init_agent_()
        # Clear History
        self.history = []

### RL Agent Helper Functions

def get_encoded_state_tailored(env):

    # Declaring the booleans
    food_is_behind_of_B = 0
    food_is_front_of_B = 0
    food_is_left_of_B = 0
    food_is_right_of_B = 0

    A_is_near = False
    A_is_behind_of_B_and_scoreB_more_than_scoreA = 0
    A_is_front_of_B_and_scoreB_more_than_scoreA = 0
    A_is_left_of_B_and_scoreB_more_than_scoreA = 0
    A_is_right_of_B_and_scoreB_more_than_scoreA = 0
    A_is_behind_of_B_and_scoreB_less_than_scoreA = 0
    A_is_front_of_B_and_scoreB_less_than_scoreA = 0
    A_is_left_of_B_and_scoreB_less_than_scoreA = 0
    A_is_right_of_B_and_scoreB_less_than_scoreA = 0

    border_is_left_of_B = 0
    border_is_right_of_B = 0
    border_is_front_of_B = 0

    # Finding the nearest food
    B_loc = [env.agentB["state"][0], env.agentB["state"][1]]
    food_locs = env.live_foodspawn_space
    rel_food_locs = [[loc[0] - B_loc[0], loc[1] - B_loc[1]] for loc in food_locs]
    food_dist_from_B = [abs(loc[0]) + abs(loc[1]) for loc in rel_food_locs]
    nearest_food_index = food_dist_from_B.index(min(food_dist_from_B))
    nearest_food_rel_loc = rel_food_locs[nearest_food_index]

    # Getting relative location of A. And proximity
    A_loc = [env.agentA["state"][0], env.agentA["state"][1]]
    rel_loc_of_A = [A_loc[0] - B_loc[0], A_loc[1] - B_loc[1]]
    if abs(rel_loc_of_A[0]) + abs(rel_loc_of_A[1]) == 2:
        A_is_near = True

    # Considering the head direction and updating booleans
    if env.agentB["head"] == 0:
        if nearest_food_rel_loc[0] > 0:
            food_is_behind_of_B = 1
        if nearest_food_rel_loc[0] < 0:
            food_is_front_of_B = 1
        if nearest_food_rel_loc[1] > 0:
            food_is_right_of_B = 1
        if nearest_food_rel_loc[1] < 0:
            food_is_left_of_B = 1
        if A_is_near:
            if env.agentB["score"] > env.agentA["score"]:
                if rel_loc_of_A[0] > 0:
                    A_is_behind_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[0] < 0:
                    A_is_front_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[1] > 0:
                    A_is_right_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[1] < 0:
                    A_is_left_of_B_and_scoreB_more_than_scoreA = 1
            if env.agentB["score"] <= env.agentA["score"]:
                if rel_loc_of_A[0] > 0:
                    A_is_behind_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[0] < 0:
                    A_is_front_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[1] > 0:
                    A_is_right_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[1] < 0:
                    A_is_left_of_B_and_scoreB_less_than_scoreA = 1
        if B_loc[0] == 0:
            border_is_front_of_B = 1
        if B_loc[1] == 0:
            border_is_left_of_B = 1
        if B_loc[1] == 7:
            border_is_right_of_B = 1

    if env.agentB["head"] == 1:
        if nearest_food_rel_loc[0] > 0:
            food_is_right_of_B = 1
        if nearest_food_rel_loc[0] < 0:
            food_is_left_of_B = 1
        if nearest_food_rel_loc[1] > 0:
            food_is_front_of_B = 1
        if nearest_food_rel_loc[1] < 0:
            food_is_behind_of_B = 1
        if A_is_near:
            if env.agentB["score"] > env.agentA["score"]:
                if rel_loc_of_A[0] > 0:
                    A_is_right_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[0] < 0:
                    A_is_left_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[1] > 0:
                    A_is_front_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[1] < 0:
                    A_is_behind_of_B_and_scoreB_more_than_scoreA = 1
            if env.agentB["score"] <= env.agentA["score"]:
                if rel_loc_of_A[0] > 0:
                    A_is_right_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[0] < 0:
                    A_is_left_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[1] > 0:
                    A_is_front_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[1] < 0:
                    A_is_behind_of_B_and_scoreB_less_than_scoreA = 1
        if B_loc[0] == 0:
            border_is_left_of_B = 1
        if B_loc[0] == 7:
            border_is_right_of_B = 1
        if B_loc[1] == 7:
            border_is_front_of_B = 1

    if env.agentB["head"] == 2:
        if nearest_food_rel_loc[0] > 0:
            food_is_front_of_B = 1
        if nearest_food_rel_loc[0] < 0:
            food_is_behind_of_B = 1
        if nearest_food_rel_loc[1] > 0:
            food_is_left_of_B = 1
        if nearest_food_rel_loc[1] < 0:
            food_is_right_of_B = 1
        if A_is_near:
            if env.agentB["score"] > env.agentA["score"]:
                if rel_loc_of_A[0] > 0:
                    A_is_front_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[0] < 0:
                    A_is_behind_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[1] > 0:
                    A_is_left_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[1] < 0:
                    A_is_right_of_B_and_scoreB_more_than_scoreA = 1
            if env.agentB["score"] <= env.agentA["score"]:
                if rel_loc_of_A[0] > 0:
                    A_is_front_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[0] < 0:
                    A_is_behind_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[1] > 0:
                    A_is_left_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[1] < 0:
                    A_is_right_of_B_and_scoreB_less_than_scoreA = 1
        if B_loc[0] == 7:
            border_is_front_of_B = 1
        if B_loc[1] == 0:
            border_is_right_of_B = 1
        if B_loc[1] == 7:
            border_is_left_of_B = 1

    if env.agentB["head"] == 3:
        if nearest_food_rel_loc[0] > 0:
            food_is_left_of_B = 1
        if nearest_food_rel_loc[0] < 0:
            food_is_right_of_B = 1
        if nearest_food_rel_loc[1] > 0:
            food_is_behind_of_B = 1
        if nearest_food_rel_loc[1] < 0:
            food_is_front_of_B = 1
        if A_is_near:
            if env.agentB["score"] > env.agentA["score"]:
                if rel_loc_of_A[0] > 0:
                    A_is_left_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[0] < 0:
                    A_is_right_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[1] > 0:
                    A_is_behind_of_B_and_scoreB_more_than_scoreA = 1
                if rel_loc_of_A[1] < 0:
                    A_is_front_of_B_and_scoreB_more_than_scoreA = 1
            if env.agentB["score"] <= env.agentA["score"]:
                if rel_loc_of_A[0] > 0:
                    A_is_left_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[0] < 0:
                    A_is_right_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[1] > 0:
                    A_is_behind_of_B_and_scoreB_less_than_scoreA = 1
                if rel_loc_of_A[1] < 0:
                    A_is_front_of_B_and_scoreB_less_than_scoreA = 1
        if B_loc[0] == 0:
            border_is_right_of_B = 1
        if B_loc[0] == 7:
            border_is_left_of_B = 1
        if B_loc[1] == 0:
            border_is_front_of_B = 1

    return [
        food_is_behind_of_B,
        food_is_front_of_B,
        food_is_left_of_B,
        food_is_right_of_B,
        A_is_behind_of_B_and_scoreB_more_than_scoreA,
        A_is_front_of_B_and_scoreB_more_than_scoreA,
        A_is_left_of_B_and_scoreB_more_than_scoreA,
        A_is_right_of_B_and_scoreB_more_than_scoreA,
        A_is_behind_of_B_and_scoreB_less_than_scoreA,
        A_is_front_of_B_and_scoreB_less_than_scoreA,
        A_is_left_of_B_and_scoreB_less_than_scoreA,
        A_is_right_of_B_and_scoreB_less_than_scoreA,
        border_is_left_of_B,
        border_is_right_of_B,
        border_is_front_of_B,
    ]


#DQN Model Architecture 

class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.linear = nn.Sequential(
                  nn.Linear(state_space_dim,64),
                  nn.ReLU(),
                  nn.Linear(64,64),
                  nn.ReLU(),
                  nn.Linear(64,32),
                  nn.ReLU(),
                  nn.Linear(32,action_space_dim)
                )

    def forward(self, x):
        x = x.to(device)
        return self.linear(x)

def choose_action_epsilon_greedy(net, state, epsilon):
    
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
                
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
        net_out = net(state)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions (this list includes all the actions but the optimal one)
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly from non_optimal_actions
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.cpu().numpy()

# Fill code for actionB 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net_loaded = torch.load("final_model.pth")

def get_actionB(random = False):
    #print(random)
    if random==True:
        return np.random.choice([0,1,2])
    else:
        state = get_encoded_state_tailored(env)
        actionB,_ = choose_action_epsilon_greedy(policy_net_loaded, state, 0)
        return actionB

config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100} # You can change during training but not during evaluation

env = Vasuki_custom(**config)

allowed_keys = {"Key.up":1,"Key.left":0,"Key.right":2}

def on_release(key):
    if str(key) == "Key.esc":
        return False

def on_press(key):
    key_str = str(key)
    
    if key_str not in allowed_keys.keys():
        pass
    else:
        take_step(actionA = allowed_keys[key_str])


choice_inp = int(input("To play with Random Agent, press 1 and To play with Trained bot, press 0:"))
    
def take_step(actionA):
    #print(choice_inp)
    actionB = get_actionB(random=choice_inp)
    action = {'actionA': actionA, 'actionB': actionB}
    rewardA, rewardB, done, info = env.step(action)
    # Rendering the enviroment to generate the simulation
    
    state = env.render()
    encoded, _ = env.encode()
    state = np.array(state, dtype=np.uint8)
    cv2.imshow("Game",state)
    cv2.waitKey(1)

#When running the code below, enter any key on the keyboard two times for the game to start and be



env.reset()
state = env.render()
cv2.imshow("Game",state)
cv2.waitKey(1)
with Listener(on_press = on_press,on_release=on_release) as L:
    L.join()

cv2.destroyAllWindows() 