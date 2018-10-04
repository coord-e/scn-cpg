from baselines.common.models import register
import tensorflow as tf
from baselines.a2c.utils import fc
import numpy as np

@register('cpg')
def cpg(num_sinusoids=16, num_hidden=16, observe_circular_ts=False, activation=tf.tanh):
    assert observe_circular_ts

    def network_fn(X):
        h = tf.layers.flatten(X)

        t = h[0][-1]

        # Nonlinear Control - CPG
        final = np.zeros(num_hidden, dtype=np.float32)
        for i in range(num_sinusoids):
            amp   = tf.get_variable('cpg_amp{}'.format(i),   [num_hidden], initializer=tf.constant_initializer(0.0))
            freq  = tf.get_variable('cpg_freq{}'.format(i),  [num_hidden], initializer=tf.constant_initializer(0.0))
            phase = tf.get_variable('cpg_phase{}'.format(i), [num_hidden], initializer=tf.constant_initializer(0.0))

            out = tf.multiply(amp, tf.sin(freq * t + phase))
            final = tf.add(final, out)

        return tf.reshape(final, (1, num_hidden)), None

    return network_fn
