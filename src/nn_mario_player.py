from ppo_base import PPOBase
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense

class NNPlayer(PPOBase):
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
        self.ACTIVATION = "relu"    # Activation function to use in the actor/critic networks
        self.HIDDEN_SIZE = 24       # Number of neurons in actor/critic network layers
        self.NUM_LAYERS = 1         # Number of layers in the actor and critic networks
        self.LEARNING_RATE = 1e-4   # Lower lr stabilises training greatly         

        self.critic = self.build_critic()
        self.actor = self.build_actor()

    def build_actor(self):
        """
        Builds Actor Network with optional layers with increasing neurons each layer
            The actor predicts an action based on the state of the game
        """
        state_input = Input(shape=self.NUM_STATE)
        advantage = Input(shape=(1,)) # Advantage is the critic predicted rewards subtracted from the actual rewards
        old_prediction = Input(shape=(self.NUM_ACTIONS,)) # Previous action predictions (probabilities)

        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION, name="actor_dense1_{}".format(self.ACTIVATION))(state_input)
        for i in range(self.NUM_LAYERS - 1):
            x = Dense(self.HIDDEN_SIZE * (i+2) , activation=self.ACTIVATION, name="actor_dense{0}_{1}".format(i+2, self.ACTIVATION))(x) 
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
        Builds Critic Network with optional layers with increasing neurons each layer 
            The critic predicts a reward based on the state of the game
        """
        state_input = Input(shape=self.NUM_STATE)

        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION, name="critic_dense1_{}".format(self.ACTIVATION))(state_input)
        for i in range(self.NUM_LAYERS - 1):
            x = Dense(self.HIDDEN_SIZE * (i+2), activation=self.ACTIVATION, name="critic_dense{0}_{1}".format(i+2, self.ACTIVATION))(x)
        out_value = Dense(1, name='critic_output')(x)
        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss='mse')
        model.summary()
        return model