from typing import Optional, Union

import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow.keras.layers import Layer, Lambda, InputLayer, Flatten, Dense, Reshape, Multiply, Add, Conv2D, Activation

class CrossCorrelationFFT(Layer):
    '''Computes the cross correlation between the feautres maps using FFT'''
    def __init__(self, max_shift_y, max_shift_x, is_add_flips=False, **kwargs):
        super().__init__(**kwargs)
        self.max_shift_y = max_shift_y
        self.max_shift_x = max_shift_x
        self.is_add_flips = is_add_flips

    def call(self, x_BHWC):
        n_features = K.int_shape(x_BHWC)[-1]

        s = K.shape(x_BHWC)

        x_BHWC = self._standartize(x_BHWC)

        # the tf/cuda 2D fft implementation works on the inner two dimensions, so we permute the spatial dimensions
        # to be the last two dimensions (from BHWC to BCHW)
        x_BCHW = tf.transpose(x_BHWC, [0, 3, 1, 2])

        fft_length = [s[1], s[2]]
        f_BCHW = tf.signal.rfft2d(x_BCHW, fft_length=fft_length)
        f_BCHW_conj = tf.math.conj(f_BCHW)

        cc_freq_domain_BCHW_list = []

        # Compute the cross correlation between each pair of of feature maps by multiplying point-wise in the frequency
        # domain (Cross-Correlation Theorem - https://mathworld.wolfram.com/Cross-CorrelationTheorem.html)
        # note that we could do this using broadcasting, which will probably be faster, but will more expensive
        # memory-wise (which is more than an issue here)
        # TODO: this is VERY heavy memory wise, can we do it efficiently in image space? (taking into account we are not
        #       interested in all shifts, but only in a relatively small window - 85x85 right now)

        for i in range(n_features):
            cc_freq_domain_BCHW_list.append(K.expand_dims(f_BCHW[:, i, :, :], 1) * f_BCHW_conj[:, i:, :, :])

        cc_image_domain_BCHW = tf.signal.irfft2d(K.concatenate(cc_freq_domain_BCHW_list, axis=1), fft_length=fft_length)

        # cropping only the relevant window size (which is in the corners of the map because we didn't call fftshift)
        tl = cc_image_domain_BCHW[:, :, :self.max_shift_y + 1, :self.max_shift_x + 1]
        tr = cc_image_domain_BCHW[:, :, :self.max_shift_y + 1, -self.max_shift_x:]
        bl = cc_image_domain_BCHW[:, :, -self.max_shift_y:, :self.max_shift_x + 1]
        br = cc_image_domain_BCHW[:, :, -self.max_shift_y:, -self.max_shift_x:]

        # h = 2*self.max_shift_y + 1, w = 2*self.max_shift_x + 1
        cc_image_domain_BChw = K.concatenate((K.concatenate((br, bl), axis=3), K.concatenate((tr, tl), axis=3)), axis=2)
        cc_image_domain_BhwC = tf.transpose(cc_image_domain_BChw, [0, 2, 3, 1])

        # the cross correlation has mirror symmetry (for infinite signals at least), for two 2D signals X&Y we have:
        # CC(X,Y)[i,j] = CC(Y,X)[-i,-j] (where the origin is in the center)
        # so we can compute the CC once and add flips instead of compute each pair of channels twice
        if self.is_add_flips:
            cc_image_domain_BhwC = K.concatenate([cc_image_domain_BhwC, cc_image_domain_BhwC[:, ::-1, ::-1, :]])

        # for some reason "compute_output_shape" is not called so we have to do it manually (worked in keras but not in tf.keras)
        cc_image_domain_BhwC.set_shape(self.compute_output_shape(K.int_shape(x_BHWC)))

        return cc_image_domain_BhwC

    def _standartize(self, x, axes=[1, 2], std_eps=1e-9):
        '''Standardize the input to have mean 0 and std 1'''
        s = K.shape(x)
        N = K.prod(K.gather(s, axes))

        x = x - K.mean(x, axis=axes, keepdims=True)
        stds = K.std(x, axis=axes, keepdims=True)
        stds = tf.where(stds < std_eps, tf.fill(K.shape(stds), np.inf), stds)
        x = x / (stds * K.sqrt(tf.cast(N, K.floatx())))
        return x

    def compute_output_shape(self, input_shape):
        n_features = input_shape[-1]
        out_features = (n_features * (n_features + 1)) // 2

        if self.is_add_flips:
            out_features *= 2

        return input_shape[0], self.max_shift_y * 2 + 1, self.max_shift_x * 2 + 1, out_features

class BiasLayer(Layer):
    '''A learnable bias layer'''
    def __init__(self, bias_shape, initializer='ones', regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.bias_shape = bias_shape
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        self.biases = self.add_weight(
            shape=self.bias_shape, initializer=self.initializer, regularizer=self.regularizer, trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return K.bias_add(x, self.biases)

    def compute_output_shape(self, input_shape):
        return input_shape

def guided_operation(x, guiding_tensor: tf.Tensor, op: Layer,
                     op_activation: Optional[Union[str, Activation]]='relu',
                     use_multiplicative_guidance=True, use_additive_guidance=True, guide_before_op=True):
    '''
    Preforms a guided operation (such as conv) on the given input x.
    The guidance is done by mapping the given guiding tensor (by a learneble FC layer) to additive biases and multipliers,
    one per features map, which modify the input before or after preforming the operation (controlled by guide_before_op),
    but before the activation.

    Given an input with shape BxHxWxC we map guiding_tensor to Bx1x1xC multipliers and Bx1x1xC biases (i.e. a single value
    for each features map/channel),
    The result is
        out[b, i, j, c] = act(guided_x[b, i, j, c])

    where:
        guided_x[b, i, j, c] = x{b, i, j, c}*multipliers{b, :, :, c} + biases{b, :, :, c}

    :param guide_before_op: controls if the guidance performed before or after the operation (but always before the
                            activation).
                            When True the flow is guide -> op -> act
                            When False the flow is op -> guide -> act

    '''
    if not guide_before_op:
        x = op(x)

    n_feautres = x.shape[-1]
    if use_multiplicative_guidance:
        multipliers = Dense(n_feautres)(guiding_tensor)
        multipliers = Reshape([1, 1, -1])(multipliers)
        multipliers = BiasLayer(n_feautres)(multipliers)
        x = Multiply()([multipliers, x])

    if use_additive_guidance:
        additive_biases = Dense(n_feautres)(guiding_tensor)
        additive_biases = Reshape([1, 1, -1])(additive_biases)
        x = Add()([x, additive_biases])

    if guide_before_op:
        x = op(x)

    if op_activation is not None:
        if isinstance(op_activation, str):
            op_activation = Activation(op_activation)
        x = op_activation(x)

    return x



class Standardize(Layer):
    '''Standardize the input to have mean 0 and std 1'''
    def __init__(self, epsilon=K.epsilon(), axes=[1, 2], return_mean_and_std=False, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.axes = axes
        self.return_mean_and_std = return_mean_and_std


    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=self.axes, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)  # epsilon to avoid dividing by zero
        x_normed = (x - mean) / std
        if self.return_mean_and_std:
            return [x_normed, mean, std]
        return x_normed

    def compute_output_shape(self, input_shape):
        if self.return_mean_and_std:
            return [input_shape, input_shape, input_shape]
        return input_shape

class Normalization(Layer):
    def __init__(self, ord='euclidean', eps=K.epsilon(), **kwargs):
        super().__init__(**kwargs)
        self.ord = ord
        self.eps = eps


    def call(self, x):
        s = K.shape(x)
        expanded_size = [-1]
        for i in range(1, s.shape[0]):
            expanded_size.append(1)

        return x / (K.reshape(tf.norm(K.batch_flatten(x), ord=self.ord, axis=1), expanded_size) + self.eps)

class CropCenter(Layer):
    '''Crops a window around the center'''
    def __init__(self, win_height, win_width, **kwargs):
        super().__init__(**kwargs)
        self.win_height = win_height
        self.win_width = win_width

    def call(self, x):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        y_start = h // 2 - self.win_height // 2
        x_start = w // 2 - self.win_width // 2
        out = x[:, y_start:y_start + self.win_height, x_start:x_start + self.win_width, :]
        # tf.keras doesn't call compute_output_shape automatically unfortunately (unlike the original keras lib),
        # so we've got to do it ourselves
        new_shape = self.compute_output_shape(K.int_shape(x))[1:]
        return Reshape(new_shape)(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.win_height, self.win_width, input_shape[-1])

