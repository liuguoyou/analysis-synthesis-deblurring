import numpy as np

import utils
from nns.analysis_nn import AnalysisNN
from nns.synthesis_nn import SynthesisNN
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class DeblurNN:
    def __init__(self, analysisNN: AnalysisNN, synthesisNN: SynthesisNN, weight_path=None):
        self.analysisNN = analysisNN
        self.synthesisNN = synthesisNN
        self.model = self.build_model()

        if weight_path:
            self.model.load_weights(weight_path)

    def build_model(self):
        blurred_image_input = Input(shape=(None, None, 3))
        analysis_out = self.analysisNN.model(blurred_image_input)
        synth_out = self.synthesisNN.model([blurred_image_input, analysis_out])

        return Model(inputs=blurred_image_input, outputs=synth_out)

    def deblur(self, image_HWC, **predict_kwargs):
        '''
        Deblurs a single image.

        :param image_HWC: the rgb image to deblur in shape (height, width, 3) shape.
                           the image should be in [0, 255] range (and not [0, 1])
        :return: the deblurred float image in [0, 255] range (and not [0, 1])
        '''
        assert image_HWC.ndim == 3, f'image_HWC bust have 3 dimensions, but got shape {image_HWC.shape}'

        return self.deblur_batch(np.expand_dims(image_HWC, 0), **predict_kwargs)[0]

    def deblur_batch(self, images_BHWC, batch_size=1, **predict_kwargs):
        '''
        Deblurs a batch of images

        :param images_BHWC: the images to deblur in shape (batch, height, width, 3) shape.
                            the images should be in 0-255 range (and not 0-1)
        :param batch_size: the prediction batch size (leave 1 to avoid OOME)
        :param predict_kwargs: other kwargs to pass to keras predict function
        :return: the deblurred float images in [0, 255] range (and not [0, 1])
        '''
        assert images_BHWC.ndim == 4, f'images_BHWC must have 4 dimensions, but got shape {images_BHWC.shape}'
        assert images_BHWC.shape[-1] == 3, f'images_BHWC must be a batch of rgb images, but got shape {images_BHWC.shape}'

        # making the input to be divisible by 2^n where n is the number of levels (downsamples), and the first level is 0
        n_levels = max(self.synthesisNN.n_levels(), self.analysisNN.n_levels())
        d = 2 ** n_levels
        padded_images_BHWC, padding = utils.pad_to_divisible(images_BHWC.astype(np.float32), d)

        # prediction the kernels (or some more abstract representation of them after the E2E training)
        kernels = self.analysisNN.predict(padded_images_BHWC, batch_size=batch_size, **predict_kwargs)

        # deblurring the images using the kernels
        padded_preds = self.synthesisNN.predict(padded_images_BHWC, kernels, batch_size=batch_size, **predict_kwargs)

        # removing the extra padding
        preds = utils.remove_padding(padded_preds, padding)

        return np.clip(preds, 0, 255)