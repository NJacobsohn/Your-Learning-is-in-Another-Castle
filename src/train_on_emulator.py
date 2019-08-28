import retro
import argparse
from action_discretizer import MarioDiscretizer

def random_actions(env, print_steps=False, n_steps=1000):
    #this starts an environment where mario does random actions on 1-2 and gets reinforced with the final score
    obs = env.reset()
    steps = 0
    test_movements = [1]
    while True:
        
        obs, rew, done, info = env.step(test_movements)
        steps += 1
        action_print = steps % n_steps == 0
        if print_steps and action_print:
            print(obs)
            print(info)
            #print(rew)
            #rint(done)

        env.render()
        if done:
            obs = env.reset()
    env.close()

def make_environment(args):
    """
    Takes in parsed arg dictionary and returns functioning retro environment
    """

    env = retro.make(
        game=args.game,
        info=args.variables, #these are the variables I tracked down as well as their location in memory
        obs_type=retro.Observations(args.observations), #0 for CNN image observation, 1 for NN numerical observation
        state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
        scenario=args.scenario,
        record=args.record)
    
    return MarioDiscretizer(env) #wraps env to only allow hand chosen inputs and input combos


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", '--game', default='SuperMarioWorld-Snes', help="This is the name of the game to learn on")
    parser.add_argument("-s", '--scenario', default="scenarios/scenario.json", help="Try out a custom scenario")
    parser.add_argument("-o", "--observations", default=0, help="Either 0 or 1, 0 for screen observation, 1 for numerical observation", type=int)
    parser.add_argument("-r", "--record", default="learning_movies/", help="Choose a directory to record the training session to")
    parser.add_argument("-v", "--variables", default="variables/powerup_relativeY_midway_positions.json", help="Path to reward variable json")
    args = parser.parse_args()

    env = make_environment(args)

    

    random_actions(env, print_steps=True, n_steps=1000)