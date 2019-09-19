import os
import retro
from action_discretizer import MarioDiscretizer
from baselines.common.retro_wrappers import AllowBacktracking

class AlgorithmBase(object):
    """
    This is the base class for algorithm classes that are callable from the command line.
    """

    def __init__(self, project_name, game, scenario, variables, observation_type, record):
        self.project_name = project_name
        self.game = game
        self.scenario = scenario
        self.variables = variables
        self.observation_type = retro.Observations(observation_type) 
        self.record_path = self._fix_record_path(record)

        self.isVectorized = False 

        self.max_episode_steps = 5000

        

    def _fix_record_path(self, record_path):
        """
        Moves the project directory in the chosen recording path
        """
        full_path = record_path + "/" + self.project_name + "/"
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        return full_path

    def make_env(self):
        """
        Creates generic environment for specific wrapping
        """
        env = retro.make(
            game=self.game,
            info=self.variables, #these are the variables I tracked down as well as their location in memory
            obs_type=self.observation_type, #0 for CNN image observation, 1 for NN numerical observation
            state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
            scenario=self.scenario,
            record=self.record_path)

        env = MarioDiscretizer(env) #wraps env to only allow hand chosen inputs and input combos

        env = AllowBacktracking(env) #allows network to backtrack if stuck, this is a preliminary test

        return env

    def run(self):
        """
        Runs the algorithm
        """
        pass

        