from algorithm_object_base import AlgorithmBase

class RandomPlayer(AlgorithmBase):
    """
    Creates a RandomPlayer with chosen arguments.
    This is one of few algorithm classes that is callable from the command line.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.env = self.make_env()

    def run(self, n_episodes=5):
        _ = self.env.reset()
        current_episode = 1
        while True:
            if current_episode > n_episodes:
                print("Episode Limit Reached!")
                break
            obs, rew, done, _info = self.env.step(self.env.action_space.sample())
            self.env.render()
            if done:
                print("Total Rewards for episode {0}: {1}".format(current_episode, rew))
                current_episode += 1
                obs = self.env.reset()      
        self.env.close()