import os
import retro
from action_discretizer import MarioDiscretizer
from retro.examples.brute import Brute, TimeLimit
from baselines.common.retro_wrappers import StochasticFrameSkip

class BrutePlayer(object):
    """
    Creates a BrutePlayer with chosen arguments.
    This is one of few algorithm classes that is callable from the command line.
    """

    def __init__(self, project_name, game, scenario, variables, observation_type, record):
        self.project_name = project_name
        self.game = game
        self.scenario = scenario
        self.variables = variables
        self.observation_type = observation_type 
        self.record_path = self._fix_record_path(record)

        self.isVectorized = False #not a ppo algorithm, this may need to be changed for parallelization of this algorithm
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
        Creates Brute specific environment
        """
        self.env = retro.make(
            game=self.game,
            info=self.variables, #these are the variables I tracked down as well as their location in memory
            obs_type=self.observation_type, #0 for CNN image observation, 1 for NN numerical observation
            state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
            scenario=self.scenario,
            record=self.record_path)

        self.env = MarioDiscretizer(self.env) #wraps env to only allow hand chosen inputs and input combos

        self.env = StochasticFrameSkip(self.env, 4, 0.25) #wraps env to randomly skip frames, cutting down on training time

        self.env = TimeLimit(self.env, max_episode_steps=self.max_episode_steps) #might remove this wrapper later
        self.brute_alg = Brute(self.env, max_episode_steps=self.max_episode_steps)

    def run(self, n_episodes=5):
        """
        Runs the Brute algorithm
        """
        if (not self.env) or (not self.brute_alg):
            self.make_env
        timestep_limit = n_episodes * self.max_episode_steps
        timesteps = 0
        self.best_rew = float('-inf')
        while True:
            acts, rew = self.brute_alg.run()
            timesteps += len(acts)

            if rew > best_rew:
                print("New best reward {0} => {1}".format(best_rew, rew))
                best_rew = rew
                self.env.unwrapped.record_movie(self.record_path + "{0}_reward.bk2".format(best_rew))
                self.env.reset()
                for act in acts:
                    self.env.step(act)
                    self.env.render() #remove this line when running on aws
                self.env.unwrapped.stop_record()

            if timesteps > timestep_limit:
                print("Timestep limit exceeded")
                break
        self.env.close()

        