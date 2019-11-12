import csv
import retro
import numpy as np
from keras import backend as K
from nn_mario_player import NNPlayer
from cnn_mario_player import CNNPlayer
from algorithm_object_base import AlgorithmBase
from baselines.common.retro_wrappers import StochasticFrameSkip

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
        self.MAX_EPISODES = 500
        self.EPOCHS = 2                 # 2 epochs seems the best. Actor doesn't benefit much from more, and critic sees the best performance increase on the first two.
        self.GAMMA = 0.80               # Used in reward scaling, 0.99 says rewards are scaled DOWN by 1%
        self.BUFFER_SIZE = 128          # Number of actions to fit the model to
        self.BATCH_SIZE = 8
        self.NUM_STATE = self.env.observation_space.shape
        self.NUM_ACTIONS = self.env.action_space.n
 
        self.LOSS_CLIPPING = 0.2
        self.ENTROPY_LOSS = 1e-3
        self.DUMMY_ACTION = np.zeros((1, self.NUM_ACTIONS))
        self.DUMMY_VALUE = np.zeros((1, 1))                 
        self.IS_IMAGE = self.observation_type.value == 0
        if self.IS_IMAGE:
            player = CNNPlayer()
        else:
            player = NNPlayer()
        self.actor = player.build_actor(self.NUM_STATE, self.NUM_ACTIONS, self.proximal_policy_optimization_loss)
        self.critic = player.build_critic(self.NUM_STATE)

    def _update_env(self):
        """
        Updates NUM_STATE and NUM_ACTIONS parameters in the case of wrappers messing with things after the fact
        """
        self.NUM_STATE = self.env.observation_space.shape
        self.NUM_ACTIONS = self.env.action_space.n

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

    def reset_env(self):
        """
        Resets Environment to prepare for next episode
        """
        print("Starting Episode {}\n".format(self.episode))
        self.observation = self.env.reset()
        self.reward_over_time[self.episode] = np.sum(np.array(self.reward)) # Saves total rewards for future printing
        self.reward = []
        self.episode += 1

    def get_action(self):
        """
        Looks at current state of the game and predicts an action to make
        Then formats action to be interpretable by retro-gym

        Returns:
                action            = index of chosen action in action space
                action_matrix     = array of 0s with len(action_space) with 1 at action index
                predicted_action  = array of probabilities with len(action_space), predicted probs for best move
        """ 
        p = self.actor.predict([
            self.observation.reshape((1,) + self.NUM_STATE), 
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

        Hi please refactor me omg I suck
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
    
    def _write_reward_history(self, verbose=1):
        episode_reward_path = self.record_path + "reward_history.csv"
        with open(episode_reward_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for episode_num, total_reward in self.reward_over_time.items():
                writer.writerow([episode_num, total_reward])
                if total_reward >= 0:
                    print("Episode {0}:\nReward: {1:0.2f}".format(episode_num, total_reward))

    def _save_architecture(self, weights=True):
        self.actor.save(self.record_path + "actor_model.hdf5")
        self.critic.save(self.record_path + "critic_model.hdf5")
        if weights:
            self.actor.save_weights(self.record_path + "actor_weights.hdf5")
            self.critic.save_weights(self.record_path + "critic_weights.hdf5")

    def run(self):
        """
        Actually runs the algorithm. Depending on your buffer size you might get some bonus episodes out of the deal
        """
        self._update_env
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

        self._save_architecture()
        self._write_reward_history()