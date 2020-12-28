import argparse
import time

from tqdm import tqdm
import numpy as np
import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import utils

from nns.deblur_nn import DeblurNN
from nns.analysis_nn import AnalysisNN, AnalysisNNConfig
from nns.synthesis_nn import SynthesisNN



def get_args():
    if 'COLUMNS' not in os.environ:
        os.environ['COLUMNS'] = '160'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_path', '-i', required=True, default=None,
        help="A path to a single image to deblur or a path to a directory to deblur (in this case, all images in "
             "the given directory will be deblurred)."
    )
    parser.add_argument(
        '--out_dir', '-o',
        default=os.path.join(os.path.dirname(__file__), 'results'),
        help="Where to save the deblurred results."
    )
    parser.add_argument(
        '--suffix', '-s', default='_deblurred',
        help="A suffix to add to the result image (before the image extension)",
    )

    parser.add_argument(
        '--analysis_weights_path', '-amp', required=False,
        default=os.path.join(os.path.dirname(__file__), 'models', 'analysis_weights.h5'),
        help="Path to the analysis model weights. The weights will be automatically downloaded if the path doesn't exists"
    )

    parser.add_argument(
        '--synthesis_weights_path', '-smp', required=False,
        default=os.path.join(os.path.dirname(__file__), 'models', 'synthesis_weights.h5'),
        help="Path to the synthesis model weights. The weights will be automatically downloaded if the path doesn't exists"
    )

    parser.add_argument(
        '--analysis_max_input_size', '-amis', nargs='+', type=int, required=False, default=None,
        help="The maximal input size to use when estimating the kernel. In case the size it is smaller than the input resolution, a window"
             " with the specified size is cropped around the center and used for the kernel prediction (this would be fine as long as the window "
             "is big enough to statistically represent the image&blur, but in general it will hurt the deblurring accuracy)\n"
             "This is useful in case we don't have enough memory in the GPU and are getting OOME (out of memory exception)."
             "In such cases, specify this argument (try -amis 512 at first (or even something larger) and then move to smaller "
             "sizes if the error still occurs)"
    )

    parser.add_argument(
        '--plot', '-p', action='store_true', required=False, default=False,
        help="Should the results also be plotted or just saved"
    )

    parser.add_argument(
        '--side_by_side', '-sbs', action='store_true', required=False, default=False,
        help="Should the result image include both the input the deblurred image (i.e. the saved image would be the blurry and deblurred images side by side)"
    )

    return parser.parse_args()


def create_output_filename(args, input_image_path):
    # splitting the path into directory and filename
    input_dir, fname = os.path.split(input_image_path)
    # splitting filename into name and extension
    fname, ext = os.path.splitext(fname)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.suffix:
        fname += args.suffix

    return os.path.join(args.out_dir, fname + ext)

def get_images_paths_to_deblur(args):
    if os.path.isfile(args.input_path):
        images_paths = [args.input_path]
    else:
        assert os.path.isdir(args.input_path), f'argument "input_path" must be a path to a valid image or directory, but {args.input_path} is not :('
        images_paths = utils.get_images_in_dir(args.input_path)

    if len(images_paths) == 0:
        raise ValueError(f"Couldn't find any images in \"{args.input_path}\"")

    return images_paths

def create_network(args):
    # download the weights if they don't exists
    utils.download_analysis_weights(args.analysis_weights_path, force=False)
    utils.download_synthesis_weights(args.synthesis_weights_path, force=False)

    analysis_config = AnalysisNNConfig()
    analysis_config.max_input_size = args.analysis_max_input_size

    return DeblurNN(
        AnalysisNN(config=analysis_config, weights_path=args.analysis_weights_path),
        SynthesisNN(weights_path=args.synthesis_weights_path)
    )

if __name__ == '__main__':
    args = get_args()

    images_paths = get_images_paths_to_deblur(args)
    deblurNN = create_network(args)

    n = len(images_paths)
    total_deblur_time = 0
    pbar = tqdm(images_paths, total=n)

    print('\n\n*********************************************************')
    print(f'Saving results to {os.path.realpath(args.out_dir)}')
    print('*********************************************************\n\n')
    for i, image_path in enumerate(pbar):
        pbar.set_description_str(f'Deblurring image {os.path.basename(image_path)}')

        blurry_image = utils.load_image(image_path)

        t = time.time()
        # the result image is a float image in range 0-255
        deblurred_image_float = deblurNN.deblur(blurry_image)
        # the first image usually takes much longer (due to libraries loading and similar stuff) so we discard its time
        if i == 0:
            total_deblur_time += time.time() - t

        deblurred_image = np.round(deblurred_image_float).astype(np.uint8)

        out_image = deblurred_image
        if args.side_by_side:
            out_image = np.concatenate([blurry_image, deblurred_image], axis=1)

        utils.save_image(create_output_filename(args, image_path), out_image)
        if args.plot:
            utils.plot_results(image_path, blurry_image, deblurred_image)

    if n > 1:
        # discarding the time of the first image, see note above
        print(f"Average deblurring time {total_deblur_time/(n-1):.2f}s per image")
