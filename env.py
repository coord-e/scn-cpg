import gym
from gym.core import Wrapper
from gym.spaces import Box
import math
import numpy as np

class CircularTimestepObserver(Wrapper):
    observe_circular_ts = True

    def __init__(self, env, t_max=2*math.pi, t_cycle=1000):
        if not isinstance(env.observation_space, Box):
            raise NotImplementedError("Use Box")

        self.t_max = t_max
        self.t_cycle = t_cycle

        low = env.observation_space.low
        low = np.append(low, [0])
        high = env.observation_space.high
        high = np.append(high, [self.t_max])
        env.observation_space = Box(low, high)

        Wrapper.__init__(self, env=env)


    def reset(self, **kwargs):
        self.ts = 0
        ob = self.env.reset(**kwargs)
        ob = np.append(ob, [0])
        return ob

    def step(self, action):
        self.ts += 1
        ob, rew, done, info = self.env.step(action)
        ob = np.append(ob, [(self.ts % self.t_cycle) / self.t_cycle + self.t_max])
        return (ob, rew, done, info)

