import argparse
from nn_mario_player import NN_Player
from brute_algorithm import BrutePlayer
from cnn_mario_player import CNN_Player
from command_line_args import create_parser
from random_mario_player import RandomPlayer

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