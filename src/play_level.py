import json
import retro
import argparse
import numpy as np
from keras.models import load_model
from action_discretizer import MarioDiscretizer


def fix_metadata(level):
    """
    This function will rewrite the entire metadata file for SuperMarioWorld-Snes located in the gym-retro library files
    This is a stupid work around for retro gym's lack of custom state functionality. Thanks OpenAI
    """
    pass


def play_level(project_name, level_file_path, episodes=1):
    actions = 17
    action_list = []
    DUMMY_ACTION = np.zeros((1, actions))
    DUMMY_VALUE = np.zeros((1, 1)) 
    env = retro.make(
        "SuperMarioWorld-Snes",
        info="variables/data.json",
        scenario="scenarios/scenario.json",
        obs_type=retro.Observations(0),
        state=level_file_path)
    env = MarioDiscretizer(env)
    model_path = "learning_movies/" + project_name + "/"
    actor = load_model(model_path+"actor_model.hdf5")
    #critic = load_model(model_path+"critic_model.hdf5")
    #begin episode
    for n in range(episodes):
        done = False
        obs = env.reset()
        while not done:
            p = actor.predict([
                obs.reshape((1, -1)), 
                DUMMY_VALUE, 
                DUMMY_ACTION])
            action = np.random.choice(actions, p=np.nan_to_num(p[0]))
            action_list.append(action)
            obs, reward, done, _ = env.step(action)
        _ = env.reset()
        for act in action_list:
            env.render(mode="human")
            _, _, _, _ = env.step(act)
#create env with actions covered
#for however many attemps 
#while not done:
#   predict action
#   record progress + rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project")
    parser.add_argument("-l", "--level")
    parser.add_argument("-e", "--episodes")
    args = parser.parse_args()

    play_level(args.project, args.level, args.episodes)