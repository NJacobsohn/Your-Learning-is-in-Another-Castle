import argparse
#from nn_mario_player import NNPlayer
#from cnn_mario_player import CNNPlayer
#from brute_mario_player import BrutePlayer
#from random_mario_player import RandomPlayer
from command_line_args import create_parser, choose_algorithm


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    alg = choose_algorithm(args)
    
    alg.run()