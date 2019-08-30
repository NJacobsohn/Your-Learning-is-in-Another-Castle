import retro
import argparse
from action_discretizer import MarioDiscretizer
from cnn_mario_player import CNN_Player
from nn_mario_player import NN_Player
from baselines.ppo2 import ppo2
from baselines.common.models import register
from baselines.common.retro_wrappers import StochasticFrameSkip
from baselines.common.vec_env import DummyVecEnv
from retro.examples.brute import Brute, update_tree, select_actions, rollout, Node, TimeLimit, Frameskip



def random_actions(env, print_steps=False, n_steps=1000):
    #this starts an environment where mario does random actions on 1-2 and gets reinforced with the final score
    obs = env.reset()
    steps = 0
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        #return obs #line for viewing and manipulating a single observation
        steps += 1
        action_print = steps % n_steps == 0
        if print_steps and action_print:
            print(obs.shape)
            print(info)
            print(rew)
            #rint(done)

        env.render()
        if done:
            obs = env.reset()
    env.close()

@register("cnn_player")
def create_cnn_player(**conv_kwargs):
    def network_fn(X, **conv_kwargs):
        return CNN_Player(X, **conv_kwargs).model
    return network_fn

def make_environment(args):
    """
    Takes in parsed arg dictionary and returns functioning retro environment
    """

    env = retro.make(
        game=args.game,
        #max_episode_steps=None,
        info=args.variables, #these are the variables I tracked down as well as their location in memory
        obs_type=retro.Observations(args.observations), #0 for CNN image observation, 1 for NN numerical observation
        state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
        scenario=args.scenario,
        record=args.record)

    env = wrap_environment(env, args.observations)

    return env

def wrap_environment(env, observation_type):
    """
    Takes in a retro gym environment and wraps it with proper wrappers dependning on the observation type
    """

    env = MarioDiscretizer(env) #wraps env to only allow hand chosen inputs and input combos
    env = StochasticFrameSkip(env, 4, 0.25)

    #env = VecEnv(env, env.observation_space, env.action_space)
    
    if args.observations == 0: #image observation
        #env = StochasticFrameSkip(env, 4, 0.25)
        pass
    else: #numerical observation
        pass

    
    return env

def brute_retro(env, max_episode_steps=8000, timestep_limit=1e8):
    #env = retro.make(game, state, use_restricted_actions=retro.Actions.DISCRETE, scenario=scenario)
    env = Frameskip(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    brute = Brute(env, max_episode_steps=max_episode_steps)
    timesteps = 0
    best_rew = float('-inf')
    while True:
        acts, rew = brute.run()
        timesteps += len(acts)

        if rew > best_rew:
            print("new best reward {} => {}".format(best_rew, rew))
            best_rew = rew
            env.unwrapped.record_movie("best.bk2")
            env.reset()
            for act in acts:
                env.step(act)
                env.render()
            env.unwrapped.stop_record()

        if timesteps > timestep_limit:
            print("timestep limit exceeded")
            break    

def test_vec_env_builder():

    env = retro.make(
        game='SuperMarioWorld-Snes',
        #max_episode_steps=None,
        info="variables/powerup_relativeY_midway_positions.json", #these are the variables I tracked down as well as their location in memory
        obs_type=retro.Observations(0), #0 for CNN image observation, 1 for NN numerical observation
        state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
        scenario="scenarios/scenario.json",
        record="learning_movies/")  

    return env  


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", '--game', default='SuperMarioWorld-Snes', help="This is the name of the game to learn on")
    parser.add_argument("-s", '--scenario', default="scenarios/scenario.json", help="Try out a custom scenario")
    parser.add_argument("-o", "--observations", default=0, help="Either 0 or 1, 0 for screen observation, 1 for numerical observation", type=int)
    parser.add_argument("-r", "--record", default="learning_movies/", help="Choose a directory to record the training session to")
    parser.add_argument("-v", "--variables", default="variables/powerup_relativeY_midway_positions.json", help="Path to reward variable json")
    args = parser.parse_args()


    #env = DummyVecEnv([test_vec_env_builder])

    env = make_environment(args)

    brute_retro(env)

    #ppo2.learn(network="lstm", env=env, total_timesteps=10e6)

    

    #random_actions(env, print_steps=True, n_steps=100)