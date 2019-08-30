from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, SeparableConv2D
from keras.layers.convolutional import Conv2D
import os
import pickle
import numpy as np

"""
from baselines.common.models import register
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch

#define your network. this is the nature CNN with tf.nn.leaky_relu instead of relu
def custom_cnn(unscaled_images, **conv_kwargs):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.leaky_relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

#register your network
@register("custom_cnn")
def your_network_define(**conv_kwargs):
    def network_fn(X):
        return custom_cnn(X, **conv_kwargs)
    return network_fn
    
#pass the network to arguments
ppo2.learn(network='custom_cnn',...)
"""

class CNN_Player(object):
    """
    This is a player that learns how to play mario based on 
    the literal image of what the screen shows + reinforcement rewards

    *** This network ONLY learns from the screen. It doesn't recieve positional inputs ***
    """

    def __init__(self, images, **conv_kwargs):
        self.build_model(**conv_kwargs)
        self.images = images

    def build_model(self, **conv_kwargs):

        activation_func = "relu"

        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), padding='valid', input_shape=(224, 256, 3), name="conv1_b1")) #first conv. layer
        self.model.add(Activation(activation_func, name="act1_b1"))

        self.model.add(Conv2D(64, (3, 3), padding='same', name="conv2_b1")) #2nd conv. layer 
        self.model.add(Activation(activation_func, name="act2_b1"))

        return self.model

        #self.model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_b1")) # decreases size, helps prevent overfitting
        #self.model.add(Dropout(dropout_perc, name="dropout_b1")) # zeros out some fraction of inputs, helps prevent overfitting

"""
conv_kwargs = {
    "image_size" : (224, 256, 1),
    "nb_filters" : 64,
    "filter_size" : (3, 3),
}
"""


