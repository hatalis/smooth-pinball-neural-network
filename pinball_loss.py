"""
@author: Kostas Hatalis
"""
import numpy as np
import tensorflow as tf
from keras import backend as K
K.set_floatx('float64')

# pinball loss function with penalty
def pinball_loss(y, q, tau, alpha = 0.001, kappa = 0, margin = 0):
    """
    :param y: target
    :param q: predicted quantile
    :param tau: coverage level
    :param alpha: smoothing parameter
    :param kappa: penalty term
    :param margin: margin for quantile cross-over
    :return: quantile loss
    """
    
    # calculate smooth pinball loss function
    error = (y - q)
    quantile_loss = K.mean(tau * error + alpha * K.softplus(-error / alpha) )

    # calculate smooth cross-over penalty
    diff = q[:, 1:] - q[:, :-1]
    penalty = kappa * K.mean(tf.square(K.maximum(tf.Variable(tf.zeros([1], dtype=tf.float64)), margin - diff)))

    return quantile_loss + penalty

