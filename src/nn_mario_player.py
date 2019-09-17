import retro
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD #SGD isn't implemented yet but I would love to try it


from algorithm_object_base import AlgorithmBase
from action_discretizer import MarioDiscretizer
from baselines.common.retro_wrappers import StochasticFrameSkip

"""
    Below this is a version of tensorboard that exists for non-tf libraries like pytorch.
    I'm not sure why it was used here as opposed to regular tensorboard as it works with keras.
    I'm keeping it as is until I want some training visualizations, then I'm porting it over to regular tensorboard
"""
from tensorboardX import SummaryWriter 

class NNPlayer(AlgorithmBase):
    """
    This is a player that learns how to play mario based on the values in the game
    (position, coins, score, lives, etc.) + reinforcement rewards

    This uses an implementation of PPO that I built based off of various other implementations I've seen
    I built my own rather than using someone else's because I couldn't find anyone that not only used retrogym,
    but also had keras model functionality
    
    *** This network does NOT see the screen or learn off of the screen at any time ***
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.observation_type = retro.Observations(1)  # Must be 1 for numerical observation

        self.env = self.make_env()
        self.env = StochasticFrameSkip(self.env, n=4, stickprob=0.5) # Wraps env to randomly (stickprob) skip frames (n), cutting down on training time    
        
        self.episode = 0                # Current episode
        self.observation = self.env.reset()
        self.FORCE_MAX_REWARD = False                # Boolean denoting if rewards should be maximized based on predictions
        self.reward = []
        self.reward_over_time = {}
        self.writer = SummaryWriter(self.record_path)
        self.gradient_steps = 0

        self.MAX_EPISODES = 500     # Number of episodes to train over

        self.LOSS_CLIPPING = 0.2    # Only implemented clipping for the surrogate loss, paper said it was best
        self.EPOCHS = 10            # Number of Epochs to optimize on between episodes
        self.ACTIVATION = "tanh"    # Activation function to use in the actor/critic networks

        self.GAMMA = 0.99           # Used in reward scaling, 0.99 says rewards are scaled DOWN by 1%
        self.BUFFER_SIZE = 4096     # Number of actions to use in an analysis (I think)
        """
                                    The following is my train of thought as I'm working through understanding PPO and whatnot
                                    For buffer size, I think a larger number is better for training. I'm interpreting this as the number
                                    of actions the actor network will generate for the critic network to attempt to evaluate the reward of.
                                    Thus, the more actions it can accurately predict on, the more suited the network is for quickly learning
                                    new levels or challenges. The argument for a smaller buffer size is it could teach the network certain
                                    quick, easy, repeatable actions that are universally applicable to levels.
        """
        self.BATCH_SIZE = 64        # Batch size when fitting network. Smaller batch size = more weight updates.
                                    # Batch size should be both < BUFFER_SIZE and a factor of BUFFER_SIZE
        self.NUM_ACTIONS = 17       # Total number of actions in the action space
        self.NUM_STATE = 141312     # Total number of inputs from the environment (i.e. the observation space) This value is numerical observations
        self.HIDDEN_SIZE = 24       # Number of neurons in actor/critic network layers (6144 is a factor of 141312)
        self.NUM_LAYERS = 3         # Number of layers in the agent and critic networks
        self.ENTROPY_LOSS = 1e-3    # Variable in loss function, helps loss scale properly (I think)
        self.LEARNING_RATE = 1e-4   # Lower lr stabilises training greatly

        self.DUMMY_ACTION = np.zeros((1, self.NUM_ACTIONS)) # Creates array with shape (1, len(action_space))
        self.DUMMY_VALUE = np.zeros((1, 1))                 # Creates array with shape (1, 1)
                                                            # These are used as action/prediction placeholders 

        self.critic = self.build_critic()                   # Builds critic model
        self.actor = self.build_actor()                     # Builds actor model


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
        state_input = Input(shape=(self.NUM_STATE,)) # Input size is the len(observation_space)
        advantage = Input(shape=(1,)) # Advantage is the critic predicted rewards subtracted from the actual rewards
        old_prediction = Input(shape=(self.NUM_ACTIONS,)) # Previous action predictions (probabilities)

        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION)(state_input) # Add dense layer with input of correct size
        for i in range(self.NUM_LAYERS - 1): # Iterate to add network layers
            x = Dense(self.HIDDEN_SIZE * (i+1) , activation=self.ACTIVATION)(x) # Output of previous layer is input of new layers

        out_actions = Dense(self.NUM_ACTIONS, activation='softmax', name='output')(x)
        # Output later to pick an action from the action space

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE),
                      loss=[self.proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self):

        state_input = Input(shape=(self.NUM_STATE,)) # Input size is the len(observation_space)
        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION)(state_input) # Add dense layer with input of correct size
        for i in range(self.NUM_LAYERS - 1): # Iterate to add network layers
            x = Dense(self.HIDDEN_SIZE * (i+1), activation=self.ACTIVATION)(x) # Output of previous layer is input of new layers

        out_value = Dense(1)(x) # Predict reward

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
        self.reward_over_time[self.episode] = np.sum(np.array(self.reward))
        self.reward = []


    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, self.NUM_STATE), self.DUMMY_VALUE, self.DUMMY_ACTION]) # Shapes inputs to make action prediction
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
        if self.FORCE_MAX_REWARD is True:
            self.writer.add_scalar('Forced Max Episode Reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode Reward', np.array(self.reward).sum(), self.episode)
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
                #     observation       = array of len(observation_space) of current numerical state
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
            self.writer.add_scalar('Actor Loss', actor_loss.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Critic Loss', critic_loss.history['loss'][-1], self.gradient_steps)

            self.gradient_steps += 1
        for episode_num, total_reward in self.reward_over_time.items():
            if total_reward > 250:
                print("Episode {0}:\nReward: {1:0.2f}".format(episode_num, total_reward)) # This should print good episodes