#import retro
import argparse
#from baselines.ppo2 import ppo2
from nn_mario_player import NN_Player
from brute_algorithm import BrutePlayer
from cnn_mario_player import CNN_Player
from random_mario_player import RandomPlayer
#from baselines.common.models import register
#from action_discretizer import MarioDiscretizer
#from baselines.common.vec_env import DummyVecEnv, VecEnv
#from baselines.common.retro_wrappers import StochasticFrameSkip




"""def random_actions(env, print_steps=False, n_steps=1000):
    #this starts an environment where mario does random actions on 1-2 and gets reinforced with the final score
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()"""

"""@register("cnn_player")
def create_cnn_player(**conv_kwargs):
    def network_fn(X, **conv_kwargs):
        return CNN_Player(X, **conv_kwargs).model
    return network_fn"""

"""def make_environment(args):
    \"""
    Takes in parsed arg dictionary and returns functioning retro environment
    \"""

    env = retro.make(
        game=args.game,
        #max_episode_steps=None,
        info=args.variables, #these are the variables I tracked down as well as their location in memory
        obs_type=retro.Observations(args.observations), #0 for CNN image observation, 1 for NN numerical observation
        state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
        scenario=args.scenario,
        record=args.record)

    env = wrap_environment(env, args.observations)

    return env"""

"""def wrap_environment(env, observation_type=0):
    \"""
    Takes in a retro gym environment and wraps it with proper wrappers dependning on the observation type
    \"""

    env = MarioDiscretizer(env) #wraps env to only allow hand chosen inputs and input combos
    env = StochasticFrameSkip(env, 4, 0.25)

    #env = VecEnv(env, env.observation_space, env.action_space)

    if args.observations == 0: #image observation
        #env = StochasticFrameSkip(env, 4, 0.25)
        pass
    else: #numerical observation
        pass

    
    return env"""

"""def brute_retro(env, max_episode_steps=5000, timestep_limit=25000):
    env = Frameskip(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

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
            break    """

"""def test_vec_env_builder():

    env = retro.make(
        game='SuperMarioWorld-Snes',
        info="variables/powerup_relativeY_midway_positions.json", #these are the variables I tracked down as well as their location in memory
        obs_type=retro.Observations(0), #0 for image observation, 1 for numerical observation
        state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
        scenario="scenarios/scenario.json",
        record="learning_movies/")

    env = wrap_environment(env)

    return env"""  


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", default="my_project",
        help="""Define the name of your project here. 
                This will create a directory and save
                training records/models/weights to it.
                """)
    parser.add_argument("-g", '--game', default='SuperMarioWorld-Snes', help="This is the name of the game to learn on", type=str)
    parser.add_argument("-s", '--scenario', default="scenarios/scenario.json", help="Try out a custom scenario", type=str)
    parser.add_argument("-o", "--observations", default=0, help="Either 0 or 1, 0 for screen observation, 1 for numerical observation", type=int)
    parser.add_argument("-r", "--record", default="learning_movies/", help="Choose a directory to record the training session to", type=str)
    parser.add_argument("-v", "--variables", default="variables/powerup_relativeY_midway_positions.json", help="Path to reward variable json", type=str)
    parser.add_argument("-a", "--algorithm", default="brute", 
        help="""Pick name of algorithm to run. 
                Current options are:
                    Brute
                    Random

                Planned options:
                    PPO1
                    PPO2
                """)
    parser.add_argument("-m", "--model", default=None, 
        help="""Pick type of model to run with chosen algorithm.
                This argument is only necessary when PPO1/2 is chosen for algorithm
                Current options are:
                    None

                Planned options:
                    CNN
                    NN
                    LSTM?
                    XGBoost?
                """)
    args = parser.parse_args()

    if args.algorithm.lower() == "brute":
        alg = BrutePlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)

    elif args.algorithm.lower() == "random":
        alg = RandomPlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)


    alg.run()

    
    

    #env = DummyVecEnv([test_vec_env_builder]) #vectorizes environment for parallel computing/envs. MUST BE DONE FOR PPO
    #ppo2.learn(network="cnn", env=env, total_timesteps=5000)

    

    #random_actions(env, print_steps=True, n_steps=100)