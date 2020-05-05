import gym
from gym.spaces import Box
from gym.spaces import Discrete

from tqdm import trange

import numpy as np
import random
import os
import pickle

import logging
logger = logging.getLogger(__name__)

from .utils import random_maze, random_shape_maze, simple_map, solvable_map, VonNeumannMotion

SCRIPTH_PATH = os.path.dirname(os.path.realpath(__file__))

class Reward:
    ITEM = 1
    GOAL = 0
    STEP = -0.01
    INVALID = -0.01 # same as step

class Index:
    WALL = 0
    AGENT = 1
    ITEM = 2

class BaseGridWorld(gym.Env):
    def _to_impassable(self, current_map):
        return current_map == 1

    def _is_valid(self, current_map, agent):
        nonnegative = agent[0] >= 0 and agent[1] >= 0
        within_edge = agent[0] < current_map.shape[0] and agent[1] < current_map.shape[1]
        passable = not self._to_impassable(current_map)[agent[0]][agent[1]]
        return nonnegative and within_edge and passable
    
    def _is_item(self, items, agent):
        for item in items:
            if agent[0] == item[0] and agent[1] == item[1]:
                return True
        return False

class StatelessGridWorld(BaseGridWorld):
    def __init__(self):
        super().__init__()
        self.motions = VonNeumannMotion()
        
    def state_step(self, state, action):
        # Load current state
        agents = np.array(np.where(state[Index.AGENT] == 1)).transpose(1,0)
        agent = agents[0]
        items = np.array(np.where(state[Index.ITEM] == 1)).transpose(1,0)
        current_map = state[Index.WALL]

        # If done
        if len(items) == 0:
            logger.warning("You are calling 'step()' even though this environment has already returned done = True. \
                You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            return state, 0, True, {}
        
        # Do state step
        motion = self.motions[action]
        new_position = [agent[0] + motion[0], agent[1] + motion[1]]
        valid = self._is_valid(current_map, new_position)
        if valid:
            agent = new_position
        
        # Decide on rewards
        done = False
        if self._is_item(items, new_position):
            reward = Reward.ITEM
            if len(items) == 1: # the last one taken
                reward += Reward.GOAL
                done = True
        elif not valid:
            reward = Reward.INVALID
        else:
            reward = Reward.STEP
        
        # Create next state
        next_state = state.copy()
        next_state[Index.AGENT,:,:] = 0
        next_state[Index.AGENT, agent[0], agent[1]] = 1
        next_state[Index.ITEM,:,:] = 0
        for item in items:
            if agent[0] == item[0] and agent[1] == item[1]:
                continue # item is already collected
            next_state[Index.ITEM, item[0], item[1]] = 1
        
        return next_state, reward, done, {}

class GridWorldEnv(StatelessGridWorld):
    def __init__(self, height=7, width=15, n_items=4, map_type='simple', predefined=False):
        super().__init__()
        self.width = width
        self.height = height
        self.n_items = n_items
        self.observation_space = Box(low=0, high=1, shape=[3]+[self.height, self.width], dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.map_type = map_type
        self.predefined = predefined
        if self.predefined:
            with open(self.filepath, 'rb') as f:
                self.maps = pickle.load(f)
        self.reset()

    @property
    def name(self):
        return '{}_{}x{}_n{}'.format(self.map_type, self.width, self.height, self.n_items)

    @property
    def filepath(self):
        return os.path.join(SCRIPTH_PATH, 'data', '{}.pickle'.format(self.name))

    def save(self, k):
        maps = [self.generate_map() for _ in trange(k)]
        with open(self.filepath, 'wb') as f:
            pickle.dump(maps, f)

    def step(self, action):
        self.state, reward, done, info = self.state_step(self.state, action)
        return self.state, reward, done, info

    def generate_map(self):
        if self.predefined:
            return random.choice(self.maps)
        elif self.map_type == 'simple':
            return simple_map
        elif self.map_type == 'random_maze':
            return solvable_map(random_maze, self.width, self.height, complexity=0.08, density=0.1)
        elif self.map_type == 'random_shape_maze':
            return solvable_map(random_shape_maze, self.width, self.height, max_shapes=8, max_size=2, allow_overlap=False)
        else:
            raise NotImplementedError

    def reset(self):
        self.state = np.zeros((3, self.height, self.width))
        generated_map = self.generate_map()
        self.state[Index.WALL, :, :] = np.array(generated_map)

        empty = list(zip(*np.where(self.state[Index.WALL] == 0)))
        empty_positions = random.sample(empty, 1 + self.n_items)

        self.state[Index.AGENT, empty_positions[0][0], empty_positions[0][1]] = 1
        for i in range(1, len(empty_positions)):
            self.state[Index.ITEM, empty_positions[i][0], empty_positions[i][1]] = 1
        return self.state

    def render(self):
        return np.sum([(i+1)*k for i, k in enumerate(self.state)], axis=0)


def play(env):
    state = env.reset()
    print(env.render())
    rewards = 0
    for i in range(500):
        action = {'w': 0, 's': 1, 'a': 2, 'd': 3}[input()] # W,S,A,D -> up,down,left,right
        # action = 0
        state, reward, done, info = env.step(action)
        rewards += reward
        print('Action:', ['up', 'down', 'left', 'right'][action])
        print(env.render())
        print('Reward:', reward, '\tDone:', done, '\tTotal Rewards:', rewards)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GridWorld')
    parser.add_argument('--map', default='simple', type=str)
    parser.add_argument('--width', default=15, type=int)
    parser.add_argument('--height', default=7, type=int)
    parser.add_argument('--n_items', default=4, type=int)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--k', default=10000, type=int)

    args = parser.parse_args()

    env = GridWorldEnv(args.height, args.width, args.n_items, args.map)

    if args.save:
        env.save(args.k)
        exit()
    play(env)