import argparse

def create_parser():
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
                
    return parser