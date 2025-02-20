#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
# from scipy.misc import imread
from imageio import imread
from PIL import Image, JpegImagePlugin
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
import glob


def create_folder(directory):
    os.makedirs(directory, exist_ok = True)

def complete_path(directory):
    return os.path.join(directory, "")


def get_exp_types():
    
    exp_types = ["Removal", "Rotation_3D", "Rotation_2D", "Translation_3D", "Scaling", "Mix", "Translation_2D"]

    return exp_types

def check_if_exp_root(exp_root_folder, folder_list = None):

    if folder_list is None:    
        folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    
    exp_types = get_exp_types()


    for f in folder_list:
        # print(f.split("/"))
        if f.split("/")[-2] in exp_types:
            return True

    return False

def read_exp(d_path):
    save_folder = complete_path(d_path)


    img_path = save_folder + "input_image.png"
    depth_path = save_folder + "depth.npy"
    mask_path = save_folder + "input_mask.png"
    bg_path = save_folder + "background_image.png"
    depth_vis_path = save_folder + "depth.png"
    transform_path = save_folder + "transform.npy"
    im_shape = save_folder + "image_shape.npy"
    
    transformed_image_path = save_folder + "transformed_image.png"
    result_path = save_folder + "result.png"
    result_ls_path = save_folder + "resized_result_ls.png"
    zero123_result_path = save_folder + "zero123/lama_followed_by_zero123_result.png"
    object_edit_result_path = save_folder + "object_edit/result_object_edit.png"
    resized_input_image = save_folder + "resized_input_image_png.png"
    resized_input_mask = save_folder + "resized_input_mask_png.png"
    dragon_diffusion_result_path = save_folder + "dragon_diffusion/result_dragon_diffusion.png"

    
    all_paths = [img_path, depth_path, mask_path, bg_path, depth_vis_path, transform_path, transformed_image_path, result_path, im_shape, result_ls_path, zero123_result_path, resized_input_image, object_edit_result_path, resized_input_mask, dragon_diffusion_result_path]
    
    out_dict = {}
    for f_name in all_paths:
        base_name = os.path.basename(f_name)
        key_name = base_name.split(".")[0]
        f_type = base_name.split(".")[1]
        
        if file_exists(f_name):
            if f_type == "png":
                out_dict[key_name + "_png"] = read_image(f_name)
            elif f_type == "npy":
                out_dict[key_name + "_npy"] = np.load(f_name)
        else:
            out_dict[key_name + "_" + f_type] = None
    if out_dict["image_shape_npy"] is None:
        out_dict["image_shape_npy"] = np.array([512, 512])
    out_dict["path_name"] = d_path
    return out_dict

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

try:
    from .inception import InceptionV3
except ModuleNotFoundError:
    from inception import InceptionV3

# path[0] generated images, path[1] real images

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=1,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use for example --gpu 0 (leave blank for CPU only)')
parser.add_argument('--resize', default=256)
parser.add_argument('--mode', default='FID', help='set to FID or IS to calculate scores')
parser.add_argument("--method_type", default="ours", help="method type")

transform = Compose([Resize(256), CenterCrop(256), ToTensor()])
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp'}

def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False, keep_size=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        # # Official code goes below
        # images = np.array([imread(str(f)).astype(np.float32)
        #                    for f in files[start:end]])

        # # Reshape to (n_images, 3, height, width)
        # images = images.transpose((0, 3, 1, 2))
        # images /= 255
        # batch = torch.from_numpy(images).type(torch.FloatTensor)
        # #

        t = transform if not keep_size else ToTensor()

        if isinstance(files[0], pathlib.PosixPath):
            images = [t(Image.open(str(f))) for f in files[start:end]]

        elif isinstance(files[0], str):
            images = [t(Image.open(f)) for f in files[start:end]]


        elif isinstance(files[0], Image.Image):
            images = [t(f) for f in files[start:end]]

        else:
            raise ValueError(f"Unknown data type for image: {type(files[0])}")

        batch = torch.stack(images)[:, :3]
        # print(batch.shape)

        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_inception_score(preds, num_splits=10):
    """Calculates the Inception Score (IS) for a set of predictions.
    Args:
        preds (np.ndarray): The predictions array of shape (N, num_classes) where N is the number of samples.
        num_splits (int): The number of splits to use for calculating the score.
    Returns:
        tuple: A tuple containing the mean and the standard deviation of the Inception Score.
    """
    scores = []
    for index in range(num_splits):
        part = preds[index * (preds.shape[0] // num_splits): (index + 1) * (preds.shape[0] // num_splits), :]
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), axis=0)))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        scores.append(np.exp(kl_div))
    return np.mean(scores), np.std(scores)

def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False, keep_size=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose, keep_size=keep_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda, edit_method):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        # path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

        path = complete_path(path)

        if edit_method == "input":
            path_relative_to_root = "resized_input_image_png.png"
        elif edit_method == "ours":
            path_relative_to_root = "resized_result_ls.png"
        elif edit_method == "zero123":
            path_relative_to_root = "zero123/lama_followed_by_zero123_result.png"
        elif edit_method == "object_edit":
            path_relative_to_root = "object_edit/result_object_edit.png"
        elif edit_method == "dragon_diffusion":
            path_relative_to_root = "dragon_diffusion/result_dragon_diffusion.png"


        if check_if_exp_root(path):
            exp_types = get_exp_types()
            files = []
            for exp in exp_types:
                if exp == "Rotation_2D" or exp == "Scaling" or exp == "Mix" or exp == "Removal":
                    continue

                exp_p = complete_path(path + exp)
                if os.path.exists(exp_p):
                    f_d = glob.glob(exp_p + "**/" + path_relative_to_root)
                    print(exp_p, " ", len(f_d))
                    files.extend(f_d)
        else:
            files = glob.glob(path + "**/" + path_relative_to_root)

        # print(files)
        print(len(files))
        # exit()
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s


def _compute_statistics_of_images(images, model, batch_size, dims, cuda, keep_size=False):
    if isinstance(images, list):  # exact paths to files are provided
        m, s = calculate_activation_statistics(images, model, batch_size,
                                               dims, cuda, keep_size=keep_size)

        return m, s

    else:
        raise ValueError


def calculate_fid_given_paths(paths, batch_size, cuda, dims, edit_method="ours"):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda, "input")
    m2, s2 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda, edit_method)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def calculate_is_given_paths(paths, batch_size, cuda, dims):
    # Calculates the IS for two paths
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    # Assume args.path[0] is the path to generated images for IS calculation
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in pathlib.Path(paths[0]).glob('*.{}'.format(ext))])
    preds = get_activations(files, model, batch_size, dims, cuda)
    
    # Convert model logits to probabilities for IS calculation
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    
    mean_is, std_is = calculate_inception_score(preds)

    return mean_is, std_is


def calculate_fid_given_images(images, batch_size, cuda, dims, use_globals=False, keep_size=False):
    if use_globals:
        global FID_MODEL  # for multiprocessing

    for imgs in images:
        if isinstance(imgs, list) and isinstance(imgs[0], (Image.Image, JpegImagePlugin.JpegImageFile)):
            pass
        else:
            raise RuntimeError('Invalid images')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    if 'FID_MODEL' not in globals() or not use_globals:
        model = InceptionV3([block_idx])
        if cuda:
            model.cuda()

        if use_globals:
            FID_MODEL = model

    else:
        model = FID_MODEL

    m1, s1 = _compute_statistics_of_images(images[0], model, batch_size,
                                        dims, cuda, keep_size=False)
    m2, s2 = _compute_statistics_of_images(images[1], model, batch_size,
                                        dims, cuda, keep_size=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode == "FID":
        fid_value = calculate_fid_given_paths(args.path,
                                              args.batch_size,
                                              args.gpu != '',
                                              args.dims,
                                              args.method_type)
        print('FID: ', fid_value)
    elif args.mode == "IS":
        mean_is, std_is = calculate_is_given_paths(args.path,
                                              args.batch_size,
                                              args.gpu != '',
                                              args.dims)
        print(f'Inception Score: {mean_is} ± {std_is}')