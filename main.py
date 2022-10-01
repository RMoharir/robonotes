import numpy as np 
import cv2 
import gym
import random

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

class MusicTheory(Env):
    
    #metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(MusicTheory, self).__init__()

        #self.action_space = spaces.Sequence(spaces.Box(0,37, shape=(38,), dtype=int))
        self.action_space = spaces.MultiBinary(38)
        self.observation_shape = (200, 38)
        self.observation_space = spaces.Sequence(spaces.Box(38,38))

        self.state = []

        self.rounds = 200

        self.collected_reward = -1


    def reset(self):
        print("reset")
        self.state = []
        return self.state

    def _get_reward(self, observation):
    
        reward = 1

        self.collected_reward += reward

        return reward

    def step(self, action):

        done = False

        obs = self.state.append(action)
        print(obs)
        reward = self._get_reward(obs)

        self.rounds -= 1
                            
        if self.rounds == 0:
            done = True
        
        self.render(action, reward)

        return obs, done

    def render(self, action, reward):
        print("Action taken: " + (str)(action))
        print("Reward at this step: " + (str)(reward))
        print("Total reward: " + (str)(self.collected_reward))
        print("Current state is: " + (str)(len(self.state)))

def main():

    env = MusicTheory()
    print(env)
    done = False
    state = env.reset()
    while not done:
        #print(state)
        action = env.action_space.sample()
        #print(action)
        state, done = env.step(action)

    env.close()

if __name__ == "__main__":
    main()