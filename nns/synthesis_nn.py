from tensorflow.keras.layers import Dense, Conv2D, Activation, Input, Flatten, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

from custom_layers import guided_operation


class SynthesisNNConfig:
    def __init__(self):
        self.n_levels = 4

        # conv block configuration
        self.conv_block_filter_size = 3
        self.conv_block_n_features = 128
        self.conv_block_size = 3
        self.activation = 'relu'

        # guiding configuration
        self.is_blind = False
        self.dense_block_size = 2
        self.use_additive_biases = True
        self.use_multiplitive_biases = True
        self.guide_before_conv=False

        # down/up sampling configuration
        self.downsample_feature_multiplier = 1
        self.upsample_filter_size = 5
        self.downsample_filter_size = 3

        # the grid size of the maximal kernel we support
        self.max_kernel_size =(85, 85)


class SynthesisNN:
    '''A fairly standard U-Net, where the convolutions layers are guided by the estimated blur kernel'''
    def __init__(self, config=SynthesisNNConfig(), weights_path=None):
        self.config = config
        self.model: Model = self.buildmodel()
        if weights_path:
            self.load_weights(weights_path)

    def n_levels(self):
        return self.config.n_levels

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, images_BHWC, kernels_BHW, **predict_kwargs):
        assert images_BHWC.ndim == 4, f'images must have 4 dims (BHWC), but got shape {images_BHWC.shape}'
        assert kernels_BHW.ndim == 3, f'kernels must have 3 dims (BHW), but got shape {kernels_BHW.shape}'
        return self.model.predict([images_BHWC, kernels_BHW], **predict_kwargs)

    def buildmodel(self):
        c = self.config

        image_input = Input(shape=(None, None, 3))

        if c.is_blind:
            inputs = [image_input]
        else:
            kernel_input = Input(shape=c.max_kernel_size)
            flatten_guidance_vec = Flatten()(kernel_input)

            inputs = [image_input, kernel_input]

        # Downsampling part
        x = image_input

        conv_block = lambda x: self._conv_block(x, flatten_guidance_vec, c.conv_block_n_features)
        downsamples_arr = []
        for i in range(c.n_levels):
            conv_block_res = conv_block(x)
            downsamples_arr.append(conv_block_res)

            x = Conv2D(c.conv_block_n_features, c.downsample_filter_size, strides=2, padding='same')(conv_block_res)

            c.conv_block_n_features = int(c.conv_block_n_features*c.downsample_feature_multiplier)

        x = conv_block(x)

        # Upsampling part
        for i in reversed(range(c.n_levels)):
            c.conv_block_n_features = int(c.conv_block_n_features/c.downsample_feature_multiplier)

            x = Conv2DTranspose(c.conv_block_n_features, c.upsample_filter_size, strides=2, padding='same', activation=c.activation)(x)

            # concatenating the results from the corresponding downsampling layer
            x = concatenate(
                [
                    x,
                    Conv2D(c.conv_block_n_features, c.conv_block_filter_size, strides=1, padding='same')(downsamples_arr[i])
                ],
                axis=-1
            )

            x = Conv2D(c.conv_block_n_features, 1, strides=1, padding='same', activation=c.activation)(x)

            x = conv_block(x)


        output = Conv2D(3, c.conv_block_filter_size, strides=1, padding='same')(x)

        return Model(inputs=inputs, outputs=output)

    def _conv_block(self, inp, guiding_vec, filters):
        c = self.config
        if not c.is_blind:
            for units in [filters] * c.dense_block_size:
                guiding_vec = Dense(units=units, activation=c.activation)(guiding_vec)

        out = inp
        for i in range(c.conv_block_size):
            conv2d = Conv2D(filters, c.conv_block_filter_size, strides=1, padding='same')
            act = Activation(c.activation)
            if not c.is_blind:
                out = guided_operation(
                    out, guiding_vec, conv2d,
                    op_activation=act,
                    use_additive_guidance=c.use_additive_biases,
                    use_multiplicative_guidance=c.use_multiplitive_biases,
                    guide_before_op=c.guide_before_conv
                )
            else:
                out = conv2d(out)
                out = act(out)

        return out
