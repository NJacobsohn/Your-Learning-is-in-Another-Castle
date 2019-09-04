import os
import retro
import argparse
from action_discretizer import MarioDiscretizer
from baselines.common.retro_wrappers import StochasticFrameSkip
from retro.examples.brute import Brute, update_tree, select_actions, rollout, Node, TimeLimit, Frameskip

class BrutePlayer(object):

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
        full_path = record_path + "/" + self.project_name + "/"
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        
        return full_path

    def make_env(self):
        self.env = retro.make(
            game=self.game,
            info=self.variables, #these are the variables I tracked down as well as their location in memory
            obs_type=self.observation_type, #0 for CNN image observation, 1 for NN numerical observation
            state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
            scenario=self.scenario,
            record=self.record_path)
        self.env = MarioDiscretizer(self.env) #wraps env to only allow hand chosen inputs and input combos
        self.env = StochasticFrameSkip(self.env, 4, 0.25)
        self.env = TimeLimit(self.env, max_episode_steps=max_episode_steps)

    def brute_retro(self, env, max_episode_steps=5000, timestep_limit=25000):
        

        brute = Brute(env, max_episode_steps=max_episode_steps)
        timesteps = 0
        best_rew = float('-inf')
        while True:
            acts, rew = brute.run()
            timesteps += len(acts)

            if rew > best_rew:
                print("new best reward {} => {}".format(best_rew, rew))
                best_rew = rew
                env.unwrapped.record_movie("/learning_movies/{0}_reward.bk2".format(best_rew))
                env.reset()
                for act in acts:
                    env.step(act)
                    env.render()
                env.unwrapped.stop_record()

            if timesteps > timestep_limit:
                print("timestep limit exceeded")
                break  