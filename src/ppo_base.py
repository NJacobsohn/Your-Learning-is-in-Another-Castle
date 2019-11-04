import retro
import numpy as np

from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Conv2D

from algorithm_object_base import AlgorithmBase
from baselines.common.retro_wrappers import StochasticFrameSkip, Downsample, Rgb2gray

class PPOBase(AlgorithmBase):
    """
    This is a base class for optimizing models with PPO
    
    This uses an implementation of PPO that I built based off of various other implementations I've seen
    I built my own rather than using someone else's because I couldn't find anyone that not only used retrogym,
    but also had keras model functionality
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.env = self.make_env()
        self.env = StochasticFrameSkip(self.env, n=4, stickprob=0.5) # Wraps env to randomly (stickprob) skip frames (n), cutting down on training time

        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward_over_time = {}
        self.actor_critic_losses = [{}, {}]

        self.MAX_EPISODES = 100         # Number of episodes to train over
        self.LOSS_CLIPPING = 0.2        # Only implemented clipping for the surrogate loss, paper said it was best
        self.EPOCHS = 10                # Number of Epochs to optimize on between episodes
        #self.ACTIVATION = "tanh"        # Activation function to use in the actor/critic networks
        self.GAMMA = 0.85               # Used in reward scaling, 0.99 says rewards are scaled DOWN by 1% (try 0.01 on this)
        self.BUFFER_SIZE = 64           # Number of actions to use in an analysis
        self.BATCH_SIZE = 8             # Batch size when fitting network. Smaller batch size = more weight updates.
                                        # Batch size should be both < BUFFER_SIZE and a factor of BUFFER_SIZE
        self.NUM_STATE = self.env.observation_space.shape
        self.NUM_ACTIONS = self.env.action_space.n           # Total number of actions in the action space
        #self.NUM_FILTERS = 8            # Preliminary number of filters for the layers in agent/critic networks
        #self.HIDDEN_SIZE = 8            # Number of neurons in actor/critic network final dense layers
        #self.NUM_LAYERS = 2             # Number of convolutional layers in the agent and critic networks
        self.ENTROPY_LOSS = 1e-3        # Variable in loss function, helps loss scale properly
        #self.LEARNING_RATE = 1e-4       # Lower lr stabilises training greatly

        # These are used as action/prediction placeholders 
        self.DUMMY_ACTION = np.zeros((1, self.NUM_ACTIONS))
        self.DUMMY_VALUE = np.zeros((1, 1))                 
        
        self.IS_IMAGE = self.observation_type.value == 0
        

        #self.critic = self.build_critic()                   
        #self.actor = self.build_actor() 

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        """
        PPO Loss Function for Actor
        """
        def loss(y_true, y_pred):
            prob = y_true * y_pred 
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)
            loss_clip = K.clip(r, min_value=1 - self.LOSS_CLIPPING, max_value=1 + self.LOSS_CLIPPING)
            inverse_prob = -(prob * K.log(prob + 1e-10))
            return -K.mean(K.minimum(r * advantage, loss_clip * advantage) + self.ENTROPY_LOSS * inverse_prob)
        return loss
    '''
    def build_actor(self):
        """
        Builds actor network specific for training, define these in classes that inherit this one

        This default model is meant to be very bad and silly, this just exists to test PPO functionality
        """
        state_input = Input(shape=(self.NUM_STATE,))
        advantage = Input(shape=(1,)) # Advantage is the critic predicted rewards subtracted from the actual rewards
        old_prediction = Input(shape=(self.NUM_ACTIONS,)) # Previous action predictions (probabilities)
        x = Dense(1, activation=self.ACTIVATION, name="actor_dense1_{}".format(self.ACTIVATION))(state_input)
        out_actions = Dense(self.NUM_ACTIONS, activation='softmax', name='actor_output')(x)
        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE),
                      loss=[self.proximal_policy_optimization_loss(
                            advantage=advantage,
                            old_prediction=old_prediction)])
        model.summary()
        return model

    def build_critic(self):
        """
        Builds critic network specific for training, define these in classes that inherit this one

        This default model is meant to be very bad and silly, this just exists to test PPO functionality
        """
        state_input = Input(shape=(self.NUM_STATE,))
        x = Dense(1, activation=self.ACTIVATION, name="critic_dense1_{}".format(self.ACTIVATION))(state_input)
        out_value = Dense(1, name='critic_output')(x)
        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss='mse')
        model.summary()
        return model
    '''

    def reset_env(self):
        """
        Resets Environment to prepare for next episode
        """
        self.episode += 1
        print("Starting Episode {}\n".format(self.episode))
        self.observation = self.env.reset()
        self.reward_over_time[self.episode] = np.sum(np.array(self.reward)) # Saves total rewards for future printing
        self.reward = []

    def get_action(self):
        """
        Looks at current state of the game and predicts an action to make
        Then formats action to be interpretable by retro-gym

        Returns:
                action            = index of chosen action in action space
                action_matrix     = array of 0s with len(action_space) with 1 at action index
                predicted_action  = array of probabilities with len(action_space), predicted probs for best move
        """
        if self.IS_IMAGE:
            obs_shape = (1, self.NUM_STATE[0], self.NUM_STATE[1], self.NUM_STATE[2])  
        else:
            obs_shape = (1, self.NUM_STATE)
        p = self.actor.predict([
            self.observation.reshape(obs_shape), 
            self.DUMMY_VALUE, 
            self.DUMMY_ACTION])
        action = np.random.choice(self.NUM_ACTIONS, p=np.nan_to_num(p[0]))
        action_matrix = np.zeros(self.NUM_ACTIONS)
        action_matrix[action] = 1 
        return action, action_matrix, p

    def transform_reward(self):
        """
        Reward Scaling to deal with PPO exploding rewards
        """
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * self.GAMMA

    def get_batch(self):
        """
        Create batch of observations, actions, and rewards to train the Actor and Critic networks on

        Returns:
                obs             = array of observations with shape (NUM_STATE, BUFFER_SIZE)
                action          = array of action matrices with shape (NUM_ACTIONS, BUFFER_SIZE)
                pred            = array of action probability predictions with shape (NUM_ACTIONS, BUFFER_SIZE)
                reward          = array of rewards for each observation with shape (BUFFER_SIZE, 1)
        """
        batch = [[], [], [], []]
        tmp_batch = [[], [], []] 
        while len(batch[0]) < self.BUFFER_SIZE:
            action, action_matrix, predicted_action = self.get_action() 
            observation, reward, done, _ = self.env.step(action)
            self.reward.append(reward) 
            tmp_batch[0].append(self.observation)   
            tmp_batch[1].append(action_matrix)      
            tmp_batch[2].append(predicted_action) 
            self.observation = observation
            if done:    # Level was either completed or Mario died
                self.transform_reward()
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
        """
        Actually runs the algorithm. Depending on your buffer size you might get some bonus episodes out of the deal
        """
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

        self.actor.save_weights(self.record_path + "actor_weights.hdf5")
        self.critic.save_weights(self.record_path + "critic_weights.hdf5")
        self.actor.save(self.record_path + "actor_model.hdf5")
        self.critic.save(self.record_path + "critic_model.hdf5")

        for episode_num, total_reward in self.reward_over_time.items():
            if total_reward > 0:
                print("Episode {0}:\nReward: {1:0.2f}".format(episode_num, total_reward))
        # Verbosity Guide:
        # > 100: prints a lot of episodes, even some where the midway point wasn't reached
        # > 200: should print most midpoint crossings (low chance to miss it)
        # > 400: should pretty much only print level completions and outliers (this is nice)