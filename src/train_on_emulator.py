import argparse
from command_line_args import create_parser, choose_algorithm


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    alg = choose_algorithm(args)
    
    #alg.run()