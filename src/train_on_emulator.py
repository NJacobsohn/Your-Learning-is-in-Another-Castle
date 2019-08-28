import retro
import argparse

def random_actions(env, print_steps=False, n_steps=1000):
    #this starts an environment where mario does random actions on 1-2 and gets reinforced with the final score
    obs = env.reset()
    steps = 0
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        steps += 1
        action_print = steps % n_steps == 0
        if print_steps and action_print:
            print(obs)

        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", '--game', default='SuperMarioWorld-Snes', help="This is the name of the game to learn on")
    parser.add_argument("-st", '--state', default=retro.State.DEFAULT, help="Pick a save state (generally a specific level) to learn")
    parser.add_argument("-sc", '--scenario', default=None, help="Try out a custom scenario")
    parser.add_argument("-o", "--observations", default=0, help="Either 0 or 1, 0 for screen observation, 1 for numerical observation", type=int)
    parser.add_argument("-r", "--record", default=False, help="Choose a directory to record the training session to")
    args = parser.parse_args()


    env = retro.make(
        game=args.game, 
        obs_type=retro.Observations(args.observations), 
        state=args.state, 
        scenario=args.scenario,
        record=args.record)

    #env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
    random_actions(env, print_steps=True, n_steps=1000)



#___________________________________ USEFUL INFORMATION ___________________________________#
"""
BUTTON INPUTS:
['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']

BUTTON COMBOS:
[[0, 16, 32],
 [0, 64, 128],
 [0, 1, 2, 3, 256, 257, 259, 512, 514, 515, 768, 770, 771],
 [0, 1024, 2048, 3072]]








"""