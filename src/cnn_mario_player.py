from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, SeparableConv2D
from keras.layers.convolutional import Conv2D
import os
import pickle
import numpy as np

class CNN_Player(object):
    """
    This is a player that learns how to play mario based on 
    the literal image of what the screen shows + reinforcement rewards

    *** This network ONLY learns from the screen. It doesn't recieve positional inputs ***
    """

    def __init__(self):
        pass
