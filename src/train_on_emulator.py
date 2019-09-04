import argparse
from nn_mario_player import NNPlayer
from cnn_mario_player import CNNPlayer
from brute_mario_player import BrutePlayer
from command_line_args import create_parser
from random_mario_player import RandomPlayer

def choose_algorithm(args):
    if args.algorithm.lower() == "brute":
        return BrutePlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)

    elif args.algorithm.lower() == "random":
        return RandomPlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)

    elif args.algorithm.lower() == "ppo":
        try:
            model_type = args.model.lower()
        except:
            return "You specified PPO but didn't specify a model to optimize"

        if model_type == "cnn": 
            return CNNPlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)
            
        elif model_type == "nn":
            return NNPlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)



if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    alg = choose_algorithm(args)
    

    alg.run()