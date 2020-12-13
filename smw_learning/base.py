import os
import gym
import retro
import numpy as np


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

    BUTTON LIST/INDICES AS INTERPRETED BY THE ENV.STEP() FUNCTION

        ['B',
         'Y',
         'SELECT',
         'START',
         'UP',
         'DOWN',
         'LEFT',
         'RIGHT',
         'A',
         'X',
         'L',
         'R']
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
            ['B', 'UP'] #jump into a ceiling pipe (I don't think this is required to beat any levels, but I want to keep things open)
            ])




class AlgorithmBase(object):
    """
    This is the base class for algorithm classes that are callable from the command line.
    """
    def __init__(self, project_name=None, game='SuperMarioWorld-Snes', state=None, 
                 scenario=None, observation_type=None, record=None, variables=None,
                 *args, **kwargs):
        """
        Args:
            project_name (str): Name of project and directory to save recordings in
            game (str): Name of ROM to load, defaults to SMW for the SNES
            state (str): Name of level (or state) to load for training
            scenario (str): Path to json file describing the environment and parameters for fitness
            observation_type (int or bool): 0 or 1. 0 for screen observations or 1 for memory state (2D vs 1D)
            record (str): Path to specific directory within project dir to save recordings to
            variables (str): Path to json with memory mapping of in-game variables
        
        """
        self.project_name = project_name
        self.game = game
        self.state = state
        self.scenario = scenario
        self.variables = variables
        if isinstance(observation_type, bool):
            observation_type = int(observation_type)
        self.observation_type = retro.Observations(observation_type) 
        self.record_path = self._fix_record_path(record)

    def _fix_record_path(self, record_path):
        """
        Moves the project directory in the chosen recording path
        """
        full_path = record_path + self.project_name + "/"
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path

    def make_env(self):
        """
        Creates generic SMW environment for specific wrapping
        """
        env = retro.make(
            game=self.game,
            info=self.variables,
            obs_type=self.observation_type,
            scenario=self.scenario,
            record=self.record_path)
        env.load_state(self.state)
        env = MarioDiscretizer(env) #wraps env to only allow hand chosen inputs and input combos
        return env

    def run(self):
        """
        Runs the algorithm
        """
        raise NotImplementedError("Please implement custom run method for your algorithm")
