import argparse
from nn_mario_player import NNPlayer
from cnn_mario_player import CNNPlayer
from brute_mario_player import BrutePlayer
from random_mario_player import RandomPlayer

def create_parser():
    """
    Creates parser. This is over here because all the text looked messy in the train_on_emulator script
    """
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
    parser.add_argument("-v", "--variables", default="variables/data.json", help="Path to reward variable json", type=str)
    parser.add_argument("-a", "--algorithm", default="random", 
        help="""Pick name of algorithm to run. 
                Current options are:
                    Brute
                    Random
                    PPO

                Planned options:
                """)
    parser.add_argument("-m", "--model", default=None, 
        help="""Pick type of model to run with chosen algorithm.
                This argument is only necessary (currently) when PPO is chosen for algorithm.

                This argument does literally nothing if PPO isn't specified. 

                Current options are:
                    CNN *non-functioning model
                    NN *non-functioning model

                Planned options:
                    LSTM?
                    XGBoost?
                """)
                
    return parser


def choose_algorithm(args):
        """
        This is a (relatively) pythonic implementation of a switch statement.

        It's technically a nested switch statement with the PPO option
        Formatting it like this makes adding more models/algorithms very easy
        """

        switch_dict = {
        "brute" : brute_alg,
        "random" : random_alg,
        "ppo" : {
            "cnn" : cnn_model,
            "nn" : nn_model
            }
        }

        try:
            model = args.model.lower()
        except:
            print("No model was chosem")

        algorithm = args.algorithm.lower()

        if algorithm == "ppo":
            func = switch_dict.get(algorithm).get(model, "No model found for PPO") #nested .get for ppo dictionary
        else:
            func = switch_dict.get(algorithm, "No algorithm of type: {0} found".format(algorithm)) 
            
        return func(args)

def brute_alg(args):
    return BrutePlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)

def random_alg(args):
    return RandomPlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)

def cnn_model(args):
    return CNNPlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)

def nn_model(args):
    return NNPlayer(args.project, args.game, args.scenario, args.variables, args.observations, args.record)