"""
Starting below this doc string is a PPO implementation for keras models I found at:
https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py

I will be taking MANY liberties to rework this code for my use case and hopefully make it
easier to use and polymorphic ideally.



THINGS THIS CODE NEEDS TO DO:

    Option to play each episode after it completes

    Plot loss/rewards/actions/etc. each episode

    Convert code to work with images

    Multithreading support?

    Adjust NN structure?

    *** FOR THE LOVE OF GOD MAKE THIS INTO AN OBJECT THAT'S EASILY CALLABLE ***
"""


# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import numpy as np

import gym
import retro

from action_discretizer import MarioDiscretizer
from baselines.common.retro_wrappers import StochasticFrameSkip, Rgb2gray, Downsample, RewardScaler

from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam

import numba as nb
from tensorboardX import SummaryWriter

def make_env():
    env = retro.make(
            game="SuperMarioWorld-Snes",
            info="variables/data.json", #these are the variables I tracked down as well as their location in memory
            obs_type=retro.Observations(1), #0 for CNN image observation, 1 for NN numerical observation
            state=retro.State.DEFAULT, #this isn't necessary but I'm keeping it for potential future customization
            scenario="scenarios/scenario.json",
            record="learning_movies/PPO_testing/")

    env = MarioDiscretizer(env) #wraps env to only allow hand chosen inputs and input combos

    env = StochasticFrameSkip(env, 4, 0.25) #wraps env to randomly skip frames, cutting down on training time

    return env

ENV = make_env()
CONTINUOUS = False

EPISODES = 25 # Number of episodes to train over

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10 # Number of Epochs to optimize on between episodes
NOISE = 0.5 # Exploration noise

GAMMA = 0.99 # Reward scaling

BUFFER_SIZE = 1024 # Number of actions to use in an analysis
"""
    For buffer size, I think a larger number is better for training. I'm interpreting this as the number
    of actions the actor network will generate for the critic network to attempt to evaluate the reward of.
    Thus, the more actions it can accurately predict on, the more suited the network is for quickly learning
    new levels or challenges. The argument for a smaller buffer size is it could teach the network certain
    quick, easy, repeatable actions that are universally applicable to levels.
"""
BATCH_SIZE = 64
NUM_ACTIONS = 15 # Total number of actions in the action space
NUM_STATE = 141312 # Total number of inputs from the environment (i.e. the observation space) This value is numerical observations
HIDDEN_SIZE = 64
NUM_LAYERS = 1 # Number of layers in the agent and critic networks
ENTROPY_LOSS = 1e-3
LEARNING_RATE = 1e-4 # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1)) # Creates array with shape (1, len(action_space)) and a (1, 1) array 


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss



class Agent:
    def __init__(self):
        self.critic = self.build_critic()
        if CONTINUOUS:
            self.actor = self.build_actor_continuous()
        else:
            self.actor = self.build_actor()

        self.env = ENV
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.name = self.get_name()
        self.writer = SummaryWriter(self.name)
        self.gradient_steps = 0

    def get_name(self):
        name = 'learning_movies/PPO_testing/'
        return name


    def build_actor(self):
        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LEARNING_RATE),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_actor_continuous(self):
        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, activation='tanh', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LEARNING_RATE),
                      loss=[proximal_policy_optimization_loss_continuous(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self):

        state_input = Input(shape=(NUM_STATE,))
        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        print("Starting Episode {}\n".format(self.episode))
        if self.episode % 100 == 0:
            self.val = True # Decide to 
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION]) # Shapes inputs to make action prediction
        if self.val:
            action = np.argmax(p[0]) # Every 100 episodes, choose the highest prob action for success
        else:
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0])) # General case is randomly choosing an action with weighted probs based on prediction
        action_matrix = np.zeros(NUM_ACTIONS) # Creates array of zeros with len(action_space)
        action_matrix[action] = 1 # Sets the chosen action to a 1 to be interpretble by retro gym
        return action, action_matrix, p
         # action is the index in the action_space 
         # action_matrix is the array of 0s with a 1 at the action index
         # p is the probability of each action being the action to maximize the reward at current timestep

    def get_action_continuous(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        if self.val:
            action = action_matrix = p[0]
        else:
            action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        return action, action_matrix, p

    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE: # This loops generates and runs actions until the done condition is met
            if CONTINUOUS:
                action, action_matrix, predicted_action = self.get_action_continuous()
            else:
                action, action_matrix, predicted_action = self.get_action()
            observation, reward, done, info = self.env.step(action) # Take the generated action
            self.reward.append(reward) # Track reward for action

            tmp_batch[0].append(self.observation) # This is the observation, numerical/image from the game
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values
            # advantage = (advantage - advantage.mean()) / advantage.std()
            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=1)
            critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=1)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

            self.gradient_steps += 1


if __name__ == '__main__':
    ag = Agent()
    ag.run()