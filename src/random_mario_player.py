import os
import retro
from action_discretizer import MarioDiscretizer

class RandomPlayer(object):

    def __init__(self, project_name, game, scenario, variables, observation_type, record):
        self.project_name = project_name
        self.game = game
        self.scenario = scenario
        self.variables = variables
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
        self.env = retro.make(
            game=self.game,
            info=self.variables, #these are the variables I tracked down as well as their location in memory
            state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
            scenario=self.scenario,
            record=self.record_path)
        self.env = MarioDiscretizer(self.env)

    def run(self, n_episodes=5):
        obs = self.env.reset()
        current_episode = 1
        while True:
            if current_episode > n_episodes:
                print("Episode Limit Reached!")
                break

            obs, rew, done, info = self.env.step(self.env.action_space.sample())
            self.env.render()

            if done:
                print("Total Rewards for episode {0}: {1}".format(current_episode, rew))
                current_episode += 1
                obs = self.env.reset()
                
        env.close()