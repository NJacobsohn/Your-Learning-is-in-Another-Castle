import retro
import argparse
import numpy as np
import keras.backend as K
from keras.layers import Input
from keras.models import load_model
from action_discretizer import MarioDiscretizer

def proximal_policy_optimization_loss(advantage, old_prediction):
    """
    PPO Loss Function for Actor
    """
    def loss(y_true, y_pred):
        prob = y_true * y_pred 
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        loss_clip = K.clip(r, min_value=1 - 0.2, max_value=1 + 0.2)
        inverse_prob = -(prob * K.log(prob + 1e-10))
        return -K.mean(K.minimum(r * advantage, loss_clip * advantage) + 1e-3 * inverse_prob)
    return loss

def play_level(project_name, level_name, episodes=1, weighted_random=False):
    num_actions = 17
    advantage = Input(shape=(1,), name="actor_advantage")
    old_prediction = Input(shape=(num_actions,), name="actor_previous_prediction")
    env = retro.make(
        "SuperMarioWorld-Snes",
        info="variables/data.json",
        scenario="scenarios/scenario.json",
        obs_type=retro.Observations(0))
    env.load_state(level_name)
    env = MarioDiscretizer(env)
    model_path = "learning_movies/" + project_name + "/"
    actor = load_model(model_path+"actor_model.hdf5", custom_objects={'loss':proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction)})
    for _ in range(episodes):
        done = False
        obs = env.reset()
        action_list = []
        while not done:
            p = actor.predict(
                obs.reshape((1,) + env.observation_space.shape))
            if weighted_random:
                action = np.random.choice(num_actions, p=np.nan_to_num(p[0]))
            else:
                action = np.argmax(p)
            action_list.append(action)
            obs, _, done, _ = env.step(action)
        _ = env.reset()
        for act in action_list:
            env.render(mode="human")
            _, _, _, _ = env.step(act)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project")
    parser.add_argument("-l", "--level", 
        help="""Level options are as follows:

        Bridges1
        Bridges2

        ChocolateIsland1
        ChocolateIsland2
        ChocolateIsland3

        DonutPlains1
        DonutPlains2
        DonutPlains3
        DonutPlains4
        DonutPlains5

        Forest1
        Forest2
        Forest3
        Forest4
        Forest5

        VanillaDome1
        VanillaDome2
        VanillaDome3
        VanillaDome4
        VanillaDome5

        YoshiIsland1
        YoshiIsland2
        YoshiIsland3
        YoshiIsland4
        """)
    parser.add_argument("-e", "--episodes", default=1, type=int)
    parser.add_argument("-r", "--random", default=False, type=bool)
    args = parser.parse_args()

    play_level(args.project, args.level, args.episodes, args.random)