import argparse
from nn_mario_player import NNPlayer
from cnn_mario_player import CNNPlayer
from brute_mario_player import BrutePlayer
from command_line_args import create_parser
from random_mario_player import RandomPlayer



"""@register("cnn_player")
def create_cnn_player(**conv_kwargs):
    def network_fn(X, **conv_kwargs):
        return CNN_Player(X, **conv_kwargs).model
    return network_fn"""

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

    parser = create_parser()
    args = parser.parse_args()

    if args.algorithm.lower() == "brute":
        alg = BrutePlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)

    elif args.algorithm.lower() == "random":
        alg = RandomPlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)


    alg.run()

    
    

    #env = DummyVecEnv([test_vec_env_builder]) #vectorizes environment for parallel computing/envs. MUST BE DONE FOR PPO
    #ppo2.learn(network="cnn", env=env, total_timesteps=5000)

    

    #random_actions(env, print_steps=True, n_steps=100)