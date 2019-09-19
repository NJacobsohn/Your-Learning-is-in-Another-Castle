"""
Some nice features that should be added to this (or maybe even a custom object should be made):

    Auto-create directory within the project directory that saves the model and weights

    Ability to load saved weights/models to check performance on other levels

    Save reward analytics / loss / any and all numbers to plot and analyze
"""
import retro
import numpy as np

from keras import backend as K
from keras.optimizers import Adam, SGD #SGD not yet implemented but I plan to try it 
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten
from keras.layers import SeparableConv2D, Conv2D, Activation #Allow for custom activation functions + Sep2D if desired

from algorithm_object_base import AlgorithmBase
from action_discretizer import MarioDiscretizer
from baselines.common.retro_wrappers import StochasticFrameSkip, Downsample, Rgb2gray, AllowBacktracking

class CNNPlayer(AlgorithmBase):
    """
    This is a player that learns how to play mario based on the current image(s) of the screen
    
    This uses an implementation of PPO that I built based off of various other implementations I've seen
    I built my own rather than using someone else's because I couldn't find anyone that not only used retrogym,
    but also had keras model functionality
    
    *** This network ONLY sees the screen to learn/predict at all times, no memory or variable data ***
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.observation_type = retro.Observations(0)  # Must be 0 for image observation

        self.IS_COLOR = True

        self.env = self.make_env()
        self.env = StochasticFrameSkip(self.env, n=4, stickprob=0.5) # Wraps env to randomly (stickprob) skip frames (n), cutting down on training time

        #The following wrappers are here to cut down on training time even further, if desired
        #self.env = Downsample(self.env, ratio=2) # Divides each side of image by 2, thus cutting down total pixels by 4x
        self.env, self.IS_COLOR = Rgb2gray(self.env), False

        
        self.episode = 0                # Current episode
        self.observation = self.env.reset()
        self.FORCE_MAX_REWARD = False   # Boolean denoting if rewards should be maximized based on predictions (I think I'm going to remove this honestly)
        self.reward = []
        self.reward_over_time = {}
        self.actor_critic_losses = [{}, {}]
        self.gradient_steps = 0

        self.MAX_EPISODES = 100         # Number of episodes to train over

        self.LOSS_CLIPPING = 0.2        # Only implemented clipping for the surrogate loss, paper said it was best
        self.EPOCHS = 10                # Number of Epochs to optimize on between episodes
        self.ACTIVATION = "tanh"        # Activation function to use in the actor/critic networks

        self.GAMMA = 0.85               # Used in reward scaling, 0.99 says rewards are scaled DOWN by 1% (try 0.01 on this)
        self.BUFFER_SIZE = 1024         # Number of actions to use in an analysis
        self.BATCH_SIZE = 64            # Batch size when fitting network. Smaller batch size = more weight updates.
                                        # Batch size should be both < BUFFER_SIZE and a factor of BUFFER_SIZE
        self.NUM_ACTIONS = 17           # Total number of actions in the action space
        if self.IS_COLOR:
            self.NUM_STATE = (224, 256, 3)  # Image size for input
        else:
            self.NUM_STATE = (224, 256, 1)
        self.NUM_FILTERS = 4            # Preliminary number of filters for the layers in agent/critic networks
        self.HIDDEN_SIZE = 8            # Number of neurons in actor/critic network final dense layers
        self.NUM_LAYERS = 1             # Number of convolutional layers in the agent and critic networks
        self.ENTROPY_LOSS = 1e-3        # Variable in loss function, helps loss scale properly (I think)
        self.LEARNING_RATE = 1e-4       # Lower lr stabilises training greatly

        self.DUMMY_ACTION = np.zeros((1, self.NUM_ACTIONS)) # Creates array with shape (1, len(action_space))
        self.DUMMY_VALUE = np.zeros((1, 1))                 # Creates array with shape (1, 1)
                                                            # These are used as action/prediction placeholders 

        self.critic = self.build_critic()                   
        self.actor = self.build_actor()                     # Builds critic/actor model


    def proximal_policy_optimization_loss(self, advantage, old_prediction): # Custom loss function
        def loss(y_true, y_pred):
            prob = y_true * y_pred 
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)

            loss_clip = K.clip(r, min_value=1 - self.LOSS_CLIPPING, max_value=1 + self.LOSS_CLIPPING)

            inverse_prob = -(prob * K.log(prob + 1e-10))
            
            return -K.mean(K.minimum(r * advantage, loss_clip * advantage) + self.ENTROPY_LOSS * inverse_prob)
        return loss

    def build_actor(self):
        state_input = Input(shape=self.NUM_STATE, name="actor_state_input") # Input size is the (256, 224, 3)
        advantage = Input(shape=(1,), name="actor_advantage") # Advantage is the critic predicted rewards subtracted from the actual rewards
        old_prediction = Input(shape=(self.NUM_ACTIONS,), name="actor_previous_prediction") # Previous action predictions (probabilities)

        x = Conv2D(filters=self.NUM_FILTERS, kernel_size=(3, 3), padding="valid", activation="relu", name="actor_conv1_relu")(state_input)

        for i in range(self.NUM_LAYERS - 1): # Add convolutional layers
            x = Conv2D(filters=self.NUM_FILTERS * (i+2), kernel_size=(3, 3), padding="same", activation="relu", name="actor_conv{0}_relu".format(i+2))(x)
        
        x = Flatten()(x)

        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION, name="actor_dense1_tanh")(x) # Add dense layer with input of correct size

        out_actions = Dense(self.NUM_ACTIONS, activation='softmax', name='actor_output')(x)
        # Output later to pick an action from the action space

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE),
                      loss=[self.proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self):
        state_input = Input(shape=self.NUM_STATE) # Input size is the len(observation_space)

        x = Conv2D(filters=self.NUM_FILTERS, kernel_size=(3, 3), padding="valid", activation="relu", name="critic_conv1_relu")(state_input)

        for i in range(self.NUM_LAYERS - 1): # Add convolutional layers
            x = Conv2D(filters=self.NUM_FILTERS * (i+1), kernel_size=(3, 3), padding="same", activation="relu", name="critic_conv{0}_relu".format(i+2))(x)

        x = Flatten()(x)

        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION, name="critic_dense1_tanh")(x) # Add dense layer with input of correct size

        out_value = Dense(1, name="critic_output")(x) # Predict reward

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        print("Starting Episode {}\n".format(self.episode))
        if self.episode % 100 == 0:
            self.FORCE_MAX_REWARD = True 
        else:
            self.FORCE_MAX_REWARD = False
        self.observation = self.env.reset()
        self.reward_over_time[self.episode] = np.sum(np.array(self.reward)) # Saves total rewards for future printing
        self.reward = []

    def get_action(self): 
        p = self.actor.predict([
            self.observation.reshape(1, self.NUM_STATE[0], self.NUM_STATE[1], self.NUM_STATE[2]), 
            self.DUMMY_VALUE, 
            self.DUMMY_ACTION]) # Shapes inputs to make action prediction

        if self.FORCE_MAX_REWARD:
            action = np.argmax(p[0]) # Every 100 episodes, choose the highest prob action for success
        else:
            action = np.random.choice(self.NUM_ACTIONS, p=np.nan_to_num(p[0])) # General case is randomly choosing an action with weighted probs based on prediction
        action_matrix = np.zeros(self.NUM_ACTIONS) # Creates array of zeros with len(action_space)
        action_matrix[action] = 1 # Sets the chosen action to a 1 to be interpretble by retro gym
        return action, action_matrix, p
            # action is the index in the action_space 
                # action_matrix is the array of 0s with a 1 at the action index
                    # p is the probability of each action being the action to maximize the reward at current timestep

    def transform_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * self.GAMMA

    def get_batch(self):
        batch = [[], [], [], []] # observations, actions, predicted rewards, and actual rewards

        tmp_batch = [[], [], []] # observations, action_matrix, predicted_action
        while len(batch[0]) < self.BUFFER_SIZE: # This loops generates and runs actions until the done condition is met

            action, action_matrix, predicted_action = self.get_action() 
                # Returns:
                #     action            = index of chosen action in action space
                #     action_matrix     = array of 0s with len(action_space) with 1 at action index
                #     predicted_action  = array of probabilities with len(action_space), predicted probs for best move

            observation, reward, done, _ = self.env.step(action) # Take the generated action
                # Returns:
                #     observation       = array of shape(observation_space) of current image state
                #     reward            = reward recieved from doing previous action
                #     done              = boolean if any done conditions are met
            self.reward.append(reward) # Track reward for action

            tmp_batch[0].append(self.observation)   # This is the observation, numerical/image from the game
            tmp_batch[1].append(action_matrix)      # Track arrays of chosen actions
            tmp_batch[2].append(predicted_action)   # Track action probabilities
            self.observation = observation          # Set current observation to be newest observation

            if done:    # Level was either completed or Mario died
                self.transform_reward()     # Scale rewards
                if not self.FORCE_MAX_REWARD:                        # Do this for all episodes EXCEPT episodes divisible by 100
                    for i in range(len(tmp_batch[0])):  # For each observation
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                                                        # Grabs observations, action matrices, and action probability predictions
                        r = self.reward[i]
                                                        # Grabs rewards for aforementioned actions
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]    # Clears tmp_batch for next episode
                self.reset_env()            # Gets env setup for another go

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
                        # Formats data into arrays for better computation, these things are BIG
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward
            # obs       = array of observations with shape (len(observation_space), BUFFER_SIZE)
            # action    = array of action matrices with shape (len(action_space), BUFFER_SIZE)
            # pred      = array of action probability predictions with shape (len(action_space), BUFFER_SIZE)
            # reward    = array of rewards for each observation with shape (BUFFER_SIZE, 1)

    def run(self):
        while self.episode < self.MAX_EPISODES :
            obs, action, pred, reward = self.get_batch()
            obs, action, pred, reward = obs[:self.BUFFER_SIZE], action[:self.BUFFER_SIZE], pred[:self.BUFFER_SIZE], reward[:self.BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=self.BATCH_SIZE, shuffle=True, epochs=self.EPOCHS, verbose=1)
            critic_loss = self.critic.fit([obs], [reward], batch_size=self.BATCH_SIZE, shuffle=True, epochs=self.EPOCHS, verbose=1)
            self.actor_critic_losses[0][self.episode] = actor_loss
            self.actor_critic_losses[1][self.episode] = critic_loss

            self.gradient_steps += 1
        for episode_num, total_reward in self.reward_over_time.items():
            if total_reward > 0:
                print("Episode {0}:\nReward: {1:0.2f}".format(episode_num, total_reward)) # This should print good episodes
        # Verbosity Guide:
        # > 100: prints a lot of episodes, even some where the midway point wasn't reached
        # > 200: should print most midpoint crossings (low chance to miss it)
        # > 400: should pretty much only print level completions and outliers (this is nice)