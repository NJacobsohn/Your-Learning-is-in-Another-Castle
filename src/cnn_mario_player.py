from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Conv2D, AvgPool2D

class CNNPlayer(object):
    """
    This is a player that learns how to play mario based on the current image(s) of the screen
    
    This uses an implementation of PPO that I built based off of various other implementations I've seen
    I built my own rather than using someone else's because I couldn't find anyone that not only used retrogym,
    but also had keras model functionality
    
    *** This network ONLY sees the screen to learn/predict at all times, no memory or variable data ***
    """
    def __init__(self):
        self.ACTIVATION = "tanh"        # Activation function to use in the actor/critic networks
        self.NUM_FILTERS = 24            # Preliminary number of filters for the layers in agent/critic networks
        self.STRIDE = (2, 2)
        self.KERNEL_SIZE = (5, 5)
        self.HIDDEN_SIZE = 32            # Number of neurons in actor/critic network final dense layer
        self.NUM_BLOCKS = 1             # Number of convolutional layers in the agent and critic networks
        # try large filters + strides + valid padding
        self.LEARNING_RATE = 1e-4       # Lower lr stabilises training greatly
        self.parameter_dict = {
            "kernel_size":self.KERNEL_SIZE,
            "activation":self.ACTIVATION,
            "strides":self.STRIDE
            }                

    def build_actor(self, NUM_STATE, NUM_ACTIONS, LOSS_FUNC):
        """
        Builds Actor Network with optional layers with increasing filters each layer
            The actor predicts an action based on the state of the game
        """
        state_input = Input(shape=NUM_STATE, name="actor_state_input")
        advantage = Input(shape=(1,), name="actor_advantage") # Advantage is the critic predicted rewards subtracted from the actual rewards
        old_prediction = Input(shape=(NUM_ACTIONS,), name="actor_previous_prediction") # Previous action predictions (probabilities)

        x = Conv2D(filters=self.NUM_FILTERS, name="actor_block0_conv0", **self.parameter_dict)(state_input)
        for i in range(self.NUM_BLOCKS): 
            x = Conv2D(filters=self.NUM_FILTERS * (i+2), name="actor_block{0}_conv0".format(i+1), **self.parameter_dict)(x)
            x = Conv2D(filters=self.NUM_FILTERS * (i+2), name="actor_block{0}_conv1".format(i+1), padding="same", **self.parameter_dict)(x)
            x = AvgPool2D(pool_size=(2, 2), name="actor_block{0}_avgpool".format(i+1))(x)        
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

        x = Conv2D(filters=self.NUM_FILTERS, name="critic_block0_conv0", **self.parameter_dict)(state_input)
        for i in range(self.NUM_BLOCKS):
            x = Conv2D(filters=self.NUM_FILTERS * (i+2), name="critic_block{0}_conv0".format(i+1), **self.parameter_dict)(x)
            x = Conv2D(filters=self.NUM_FILTERS * (i+2), name="critic_block{0}_conv1".format(i+1), padding='same', **self.parameter_dict)(x)
            x = AvgPool2D(pool_size=(2, 2), name="critic_block{0}_avgpool".format(i+1))(x)
        x = Flatten(name="critic_flatten")(x)
        x = Dense(self.HIDDEN_SIZE, activation=self.ACTIVATION, name="critic_dense1_{0}".format(self.ACTIVATION))(x) 
        out_value = Dense(1, name="critic_output")(x) # Predict reward

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss='mse')
        model.summary()
        return model