import numpy as np
import tensorflow as tf

SPP_EXP_COEFF = 1
SPP_NEG_SLOPE_COEFF = 30

def softplusplus(x):
    # exp = np.exp(x * SPP_EXP_COEFF)
    # return np.log(1 + exp) - np.log(2) + (1 / SPP_NEG_SLOPE_COEFF) * x
    exp = tf.math.exp(x * SPP_EXP_COEFF)
    two = tf.cast(2, dtype=tf.float32)
    return tf.math.log(1 + exp) - tf.math.log(two) + (1 / SPP_NEG_SLOPE_COEFF) * x


def softplusplus_derivate(x):
    # exp = np.exp(x * SPP_EXP_COEFF)
    exp = tf.math.exp(x * SPP_EXP_COEFF)
    return SPP_EXP_COEFF * exp / (1+exp) + 1 / SPP_NEG_SLOPE_COEFF


# print(softplusplus(3))
# print(softplusplus_derivate(3))