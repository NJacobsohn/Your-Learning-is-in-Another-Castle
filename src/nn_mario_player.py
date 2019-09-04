import os
import pickle
import numpy as np
from keras.models import Sequential, Model
from algorithm_object_base import AlgorithmBase
from action_discretizer import MarioDiscretizer
from keras.layers import Dense, Dropout, Activation


class NNPlayer(AlgorithmBase):
    """
    This is a player that learns how to play mario based on 
    the values in the game (position, coins, score, lives, etc.) + reinforcement rewards
    
    *** This network does NOT see the screen or learn off of the screen at any time ***
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
