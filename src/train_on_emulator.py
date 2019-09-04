import argparse
from nn_mario_player import NNPlayer
from cnn_mario_player import CNNPlayer
from brute_mario_player import BrutePlayer
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