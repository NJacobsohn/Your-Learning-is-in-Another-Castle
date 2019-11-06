from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Conv2D

class CNNPlayer(object): #try making this class a generic class and have the ppo_base import it for cnn/nn?
    """
    This is a player that learns how to play mario based on the current image(s) of the screen
    
    This uses an implementation of PPO that I built based off of various other implementations I've seen
    I built my own rather than using someone else's because I couldn't find anyone that not only used retrogym,
    but also had keras model functionality
    
    *** This network ONLY sees the screen to learn/predict at all times, no memory or variable data ***
    """
    def __init__(self):
        self.ACTIVATION = "tanh"        # Activation function to use in the actor/critic networks
        self.NUM_FILTERS = 8            # Preliminary number of filters for the layers in agent/critic networks
        self.HIDDEN_SIZE = 8            # Number of neurons in actor/critic network final dense layers
        self.NUM_LAYERS = 2             # Number of convolutional layers in the agent and critic networks
        # try large filters + strides + valid padding
        self.LEARNING_RATE = 1e-4       # Lower lr stabilises training greatly

        #self.actor = self.build_actor() 
        #self.critic = self.build_critic()                   

    def build_actor(self, NUM_STATE, NUM_ACTIONS, LOSS_FUNC):
        """
        Builds Actor Network with optional layers with increasing filters each layer
            The actor predicts an action based on the state of the game
        """
        state_input = Input(shape=NUM_STATE, name="actor_state_input")
        advantage = Input(shape=(1,), name="actor_advantage") # Advantage is the critic predicted rewards subtracted from the actual rewards
        old_prediction = Input(shape=(NUM_ACTIONS,), name="actor_previous_prediction") # Previous action predictions (probabilities)

        x = Conv2D(filters=self.NUM_FILTERS, kernel_size=(3, 3), padding="valid", activation="relu", name="actor_conv1_relu")(state_input)
        for i in range(self.NUM_LAYERS - 1): 
            x = Conv2D(filters=self.NUM_FILTERS * (i+2), kernel_size=(3, 3), padding="same", activation="relu", name="actor_conv{0}_relu".format(i+2))(x)
        x = Flatten(name="actor_flatten")(x)
        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION, name="actor_dense1_{0}".format(self.ACTIVATION))(x) 
        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='actor_output')(x)
        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE),
                      loss=[LOSS_FUNC(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()
        return model

    def build_critic(self, NUM_STATE):
        """
        Builds Critic Network with optional layers with increasing filters each layer 
            The critic predicts a reward based on the state of the game
        """
        state_input = Input(shape=NUM_STATE) # Input size is the (224, 256, 3) image

        x = Conv2D(filters=self.NUM_FILTERS, kernel_size=(3, 3), padding="valid", activation="relu", name="critic_conv1_relu")(state_input)
        for i in range(self.NUM_LAYERS - 1):
            x = Conv2D(filters=self.NUM_FILTERS * (i+2), kernel_size=(3, 3), padding="same", activation="relu", name="critic_conv{0}_relu".format(i+2))(x)
        x = Flatten(name="critic_flatten")(x)
        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION, name="critic_dense1_{0}".format(self.ACTIVATION))(x) 
        out_value = Dense(1, name="critic_output")(x) # Predict reward

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss='mse')
        model.summary()
        return model