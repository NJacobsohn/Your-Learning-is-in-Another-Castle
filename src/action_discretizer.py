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

    This action space may need to be inflated in order to keep the AI from jumping a over and over again
        I.E. Add a bunch of left/right actions that just move left/right
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[
            ['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], #2 movement directions, ability to crouch/climb with up and down if need be
            ['X', 'RIGHT'], #sprint right
            ['X', 'LEFT'], #sprint left
            ['B'], #regular jump in place
            ['A'], #spin jump (also jump off yoshi) in place
            ['RIGHT', 'B'], #regular jump right
            ['LEFT', 'B'], #regular jump left
            ['RIGHT', 'A'], #spin jump right
            ['LEFT', 'A'], #spin jump left
            ['X', 'RIGHT', 'B'], #sprint jump right
            ['X', 'LEFT', 'B'], #sprint jump left
            ['X', 'RIGHT', 'A'], #sprint spin jump right
            ['X', 'LEFT', 'A'], #sprint spin jump left
            ['B', 'UP'] #jump into a ceiling pipe (this will likely be removed unless there are levels that require it)
            ])