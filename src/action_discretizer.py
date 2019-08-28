"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
"""

import gym
import numpy as np
import retro

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class MarioDiscretizer(Discretizer):
    """
    Use Super Mario World-specific discrete actions
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[
            ['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], #2 movement directions, ability to crouch/climb with up and down if need be
            ['X', 'RIGHT'], #sprint right
            ['X', 'LEFT'], #sprint left
            ['B'], #regular jump in place
            ['A'], #spin jump (also jump off yoshi) in place
            ['RIGHT', 'B'], #jump right
            ['LEFT', 'B'], #jump left
            ['X', 'RIGHT', 'B'], #sprint jump right
            ['X', 'LEFT', 'B'], #sprint jump left
            ['X', 'RIGHT', 'A'], #sprint spin jump right
            ['X', 'LEFT', 'A'], #sprint spin jump left
            ['B', 'UP'] #jump into a ceiling pipe (this should probably be left out most of the time)
            ])



env = retro.make(game='SuperMarioWorld-Snes')
env = MarioDiscretizer(env)
print('MarioDiscretizer action_space', env.action_space)
env.close()