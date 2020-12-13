import retro
from algorithm_object_base import AlgorithmBase
from retro.examples.brute import Brute, TimeLimit
from baselines.common.retro_wrappers import StochasticFrameSkip

class BrutePlayer(AlgorithmBase):
    """
    Creates a BrutePlayer with chosen arguments.
    This is one of the algorithm classes that is callable from the command line.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.max_episode_steps = 5000
        self.env = self.make_env()
        self.env, self.brute_alg = self.make_brute(self.env)

    def make_brute(self, env):
        """
        Creates Brute specific environment
        """
        env = StochasticFrameSkip(env, 4, 0.25) #wraps env to randomly skip frames, cutting down on training time
        env = TimeLimit(env, max_episode_steps=self.max_episode_steps) #might remove this wrapper later
        brute_alg = Brute(env, max_episode_steps=self.max_episode_steps)

        return env, brute_alg

    def run(self, n_episodes=5):
        """
        Runs the Brute algorithm
        """
        timestep_limit = n_episodes * self.max_episode_steps
        timesteps = 0
        self.best_rew = float('-inf')
        while True:
            acts, rew = self.brute_alg.run()
            timesteps += len(acts)

            if rew > self.best_rew:
                print("New best reward {0} => {1}".format(self.best_rew, rew))
                self.best_rew = rew
                self.env.unwrapped.record_movie(self.record_path + "{0}_reward.bk2".format(self.best_rew))
                self.env.reset()
                for act in acts:
                    self.env.step(act)
                    self.env.render() #remove this line when running on aws
                self.env.unwrapped.stop_record()

            if timesteps > timestep_limit:
                print("Timestep limit exceeded")
                break
        self.env.close()

        