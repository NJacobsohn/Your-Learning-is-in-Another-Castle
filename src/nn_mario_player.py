from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from baselines.common.retro_wrappers import *
import os
import pickle
import numpy as np

class NN_Player(object):
    """
    This is a player that learns how to play mario based on 
    the values in the game (position, coins, score, lives, etc.) + reinforcement rewards
    
    *** This network does NOT see the screen or learn off of the screen at any time ***
    """

    def __init__(self):
        pass
