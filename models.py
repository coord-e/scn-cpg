from baselines.common.models import register
import tensorflow as tf
from baselines.a2c.utils import fc
import numpy as np

@register('scn')
def scn(num_layers=2, num_hidden=16, activation=tf.tanh):
    def network_fn(X):
        h = tf.layers.flatten(X)

        # Linear Control
        K = np.zeros(h.shape[0] * num_hidden)
        K_reshape = np.reshape(K, (h.shape[0], num_hidden))
        action_l = p.matmul(K_reshape.T, h)
        U_l = clip(action_l)

        # Nonlinear Control - MLP
        for i in range(num_layers):
            h = activation(fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2)))

        mean = tf.add(U_l, h, name='polfinal')
        return mean, None

    return network_fn

