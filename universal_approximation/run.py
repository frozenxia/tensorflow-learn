import time
import os
import argparse
import io
dir = os.path.dirname(os.path.realpath(__file__))
print(dir)



import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def univApprox(x,hidden_dim=50):
    input_dim = 1
    output_dim = 1

    with tf.variable_scope('UniversalApproximator'):
        ua_w = tf.get_variable(name='ua_w',shape=[input_dim,hidden_dim],initializer=tf.random_normal_initializer(stddev=0.1))
        ua_b = tf.get_variable(name='ua_b',shape=[hidden_dim],initializer=tf.constant_initializer(0.))

        z = tf.matmul(x,ua_w)+ua_b
        a = tf.nn.relu(z)

        ua_v = tf.get_variable(name = 'ua_v',shape=[hidden_dim,output_dim],initializer=tf.random_normal_initializer(stddev=0.1))

        z = tf.matmul(a,ua_v)

    return z

def funtion_to_appox(x):
    return tf.sin(x)



if __name__ == '__main__':
    


