import os
import pickle
import numpy as np
from keras.models import Sequential, Model
from baselines.common.models import register
from keras.layers.convolutional import Conv2D
from algorithm_object_base import AlgorithmBase
from action_discretizer import MarioDiscretizer
from keras.layers import MaxPooling2D, SeparableConv2D
from keras.layers import Dense, Dropout, Activation, Flatten
from baselines.common.retro_wrappers import StochasticFrameSkip, Rgb2gray, Downsample, RewardScaler


class CNNPlayer(AlgorithmBase):
    """
    This is a player that learns how to play mario based on 
    the literal image of what the screen shows + reinforcement rewards

    *** This network ONLY learns from the screen. It doesn't recieve positional inputs ***
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_model(**kwargs)
        #self.images = images

    def build_model(self, **kwargs):

        activation_func = "relu"

        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), padding='valid', input_shape=(224, 256, 3), name="conv1_b1")) #first conv. layer
        self.model.add(Activation(activation_func, name="act1_b1"))

        self.model.add(Conv2D(64, (3, 3), padding='same', name="conv2_b1")) #2nd conv. layer 
        self.model.add(Activation(activation_func, name="act2_b1"))

        return self.model

        #self.model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_b1")) # decreases size, helps prevent overfitting
        #self.model.add(Dropout(dropout_perc, name="dropout_b1")) # zeros out some fraction of inputs, helps prevent overfitting

    def make_env(self):
        pass

    def run(self):
        pass

"""
conv_kwargs = {
    "image_size" : (224, 256, 1),
    "nb_filters" : 64,
    "filter_size" : (3, 3),
}
"""
@register("cnn_player")
def create_cnn_player(**kwargs):
    def network_fn(X, **kwargs):
        return CNNPlayer(X, **kwargs).model
    return network_fn

