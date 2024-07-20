from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from gradio_new import sample_model, load_model_from_config, preprocess_image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
from torchvision import transforms
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from contextlib import nullcontext
from omegaconf import OmegaConf
import torch
from segment_anything import SamPredictor, sam_model_registry
import cv2
from scipy.ndimage import maximum_filter
from simple_lama_inpainting import SimpleLama
from PIL import Image
import glob
from rembg import remove
from skimage.transform import resize
from skimage.filters import gaussian
import blending
import math
import argparse
import time
SAM_MODEL = None




def initialize_models(config, ckpt, device="cuda"):

    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)
    # print('Instantiating Carvekit HiInterface...')
    # models['carvekit'] = create_carvekit_interface()
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    models['nsfw'].concept_embeds_weights *= 1.07
    models['nsfw'].special_care_embeds_weights *= 1.07
    return models

def run_zero123(models, input_im, x, y, z, h = 256, w = 256, device="cuda", scale=3.0, n_samples=1, ddim_steps=50, ddim_eta=1.0, precision='fp32'):

    precision_scope = autocast if precision == 'autocast' else nullcontext

    print(input_im.min(), input_im.max())
    input_im = torch.tensor(input_im)[None].to(device)
    input_im = input_im.permute(0, -1, 1, 2)
    # input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    print(input_im.shape, input_im.min(), input_im.max())
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(models['turncam'])
    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    used_x = x  # NOTE: Set this way for consistency.
    print("Running generation sample")

    x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                    ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)

    output_ims = []
    for x_sample in x_samples_ddim:
        print(x_sample.shape)
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(PIL.Image.fromarray(x_sample.astype(np.uint8)))


    return output_ims

    # sample_model(model)
    # return out_image



def create_folder(directory):
    os.makedirs(directory, exist_ok = True)

def complete_path(directory):
    return os.path.join(directory, "")

def read_image(im_path):

    im = plt.imread(im_path)
    # print(im.min(), im.max(), im.shape, im_path)

    if len(im.shape) == 3:
        im = im[..., :3]
    
    if im.max() <= 1.0:
        im = (im * 255.0).astype("uint8")
    # print(im.min(), im.max(), im.shape, im_path)
    return im

def file_exists(f_path):
    return os.path.exists(f_path)

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
    transformed_mask_path = save_folder + "ours/transformed_mask_square.png"
    inpainted_background_path = save_folder + "lama_result/image.png"
    
    all_paths = [img_path, depth_path, mask_path, bg_path, depth_vis_path, transform_path, transformed_image_path, result_path, im_shape, transformed_mask_path]
    
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

    if file_exists(inpainted_background_path):
        print("reading inpainted background from :", inpainted_background_path)
        out_dict["inpainted_background"] = read_image(inpainted_background_path)
    else:
        out_dict["inpainted_background"] = None

    out_dict["path_name"] = d_path
    return out_dict


def list_exp_details(exp_dict):

    for k in exp_dict.keys():
        if exp_dict[k] is None:
            print(k, " None")
        elif k != "path_name":
            print(k, " ", exp_dict[k].shape, " ", exp_dict[k].min(), " ", exp_dict[k].max())
        elif k == "path_name":
            print(k, " ", exp_dict[k])

def normalize_image(im):

    if im.max() > 1:
        return im / 255.0
    else:
        return im

def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = PIL.Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result




def get_input(input_im, mask_im):

    # print(input_im.shape, mask_im.shape)
    # print(input_im.max(), mask_im.max())

    if len(mask_im.shape) > 2:
        mask_im = mask_im[..., 0]

    mask_im = (normalize_image(mask_im) > 0.5) * 1.0
    input_im = normalize_image(input_im)

    h, w = np.indices(mask_im.shape)

    h_min, h_max = h[mask_im > 0.5].min(), h[mask_im > 0.5].max()
    w_min, w_max = w[mask_im > 0.5].min(), w[mask_im > 0.5].max()


    im_out = input_im[h_min:h_max, w_min:w_max]
    im_mask_out = mask_im[h_min:h_max, w_min:w_max]
    # input_im * mask_im[..., None]

    im_out[im_mask_out < 0.5] = 1.0
    im_out = (im_out * 255.0).astype("uint8")

    h_out, w_out, _ = im_out.shape

    # print(h_out, w_out, "before")


    image = PIL.Image.fromarray(np.array(im_out))

    # resize image such that long edge is 512
    image.thumbnail([200, 200], PIL.Image.Resampling.LANCZOS)

    h_t, w_t = image.size

    s_max = h_t
    if w_t > h_t:
        s_max = w_t

    print("Max size: ", s_max)

    if h_out > w_out:
        scale_factor = s_max / h_out
    else:
        scale_factor = s_max / w_out
    # print(image.size, "after,", scale_factor, image.size[0] / scale_factor, image.size[1] / scale_factor)
    image = add_margin(image, (255, 255, 255), size=256)
    image = np.array(image)

    image = (image / 255.0).astype(np.float32)

    return image, scale_factor

def rotateAxis(degrees, axis):
    '''
    Function to rotate around given axis

    Input:
        degrees - scalar - Angle in degrees
        
        axis - scalar - options:
            0 - around x axis
            1 - around y axis
            2 - around z axis  
    
    Returns:
        Homogeneous rotation matrix
    '''

    radians = np.radians(degrees)

    if axis == 2: # z - axis

        rotation_mat = np.array([[np.cos(radians), -np.sin(radians),           0,          0],
                                 [np.sin(radians),  np.cos(radians),           0,          0],
                                 [              0,                0,           1,          0],
                                 [              0,                0,           0,          1]])

    elif axis == 1: # y - axis

        rotation_mat = np.array([[np.cos(radians),                0,  np.sin(radians),          0],
                                 [              0,                1,                0,          0],
                                 [-np.sin(radians),               0, np.cos(radians),          0],
                                 [              0,                0,                0,          1]])

    elif axis == 0: # x - axis


        rotation_mat = np.array([[             1,                0,                0,          0],
                                [              0,  np.cos(radians), -np.sin(radians),          0],
                                [              0,  np.sin(radians),  np.cos(radians),          0], 
                                [              0,                0,                0,          1]])
    
    return rotation_mat.astype("float32")






transform_converter = rotateAxis(-90, 0) #@ rotateAxis(180, 2)

# transform_converter = rotateAxis(90, 0) @ rotateAxis(180, 2)
# transform_converter = rotateAxis(-90, 0) @ rotateAxis(180, 2)# @ rotateAxis(90, 0)
# transform_converter = np.eye(4).astype("float32")


def convert_transform_geodiff_to_zero123(transform_mat):

    return np.linalg.inv(transform_converter) @ transform_mat @ (transform_converter)


def cartesian_to_spherical(xyz):
    xyz = (transform_converter[:3, :3] @ xyz.T).T
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    # print(ptsnew.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.rad2deg(np.arctan2(np.sqrt(xy), xyz[:,2])) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.rad2deg(np.arctan2(xyz[:,1], xyz[:,0]))
    return np.array([theta, azimuth, z])

def translateMatrix(x, y, z):

    translation_matrix = np.eye(4)
    translation_matrix[0,3] += x
    translation_matrix[1,3] += y
    translation_matrix[2,3] += z

    return translation_matrix


def process_depth(depth):
    depth = depth.max() - depth
    depth = depth / (depth.max() + 1e-8)
    depth[depth > 0.9] = 1000000.0
    mask = (depth < 100.0) * 1.0

    return depth, mask
    # # depth = 1.0 / depth
    # # print(depth.min(), depth.max())

    # # Constant depth case
    # if np.sum(depth) == 0.0:
    #     depth = np.ones_like(depth) * 0.5

    
    # if obj_mask is not None:
    #     mask = obj_mask * mask


def transformation_matrix_to_spherical_2(transform_mat, depth, mask):

    r = transform_mat[:3, :3]
    print(r)
    if np.linalg.det(r) < 0:
        return None, None, None
    rt = transform_mat[:3, -1]
    # print(r)
    dist = np.sum(np.abs(r - np.eye(3))) 
    print(rt)

    x_out, y_out, z_out = 0, 0, 0

    # R = transform_mat[:3, :3]
    # Reflection case

    # print(dist)
    if dist > 1e-8:
        print("rotation only")
        t = np.array([0, 0, 1]).astype("float32")
        x, y, z = cartesian_to_spherical(t[None])
        t = np.array([0, 0, rt[-1] + 1.0]).astype("float32")
        rt = r.T @ t
        xr, yr, zr = cartesian_to_spherical(rt[None, :])

        x_out += xr - x
        y_out += yr - y
        z_out += zr - z

        return x_out, y_out, z_out


    elif (np.sum(np.abs(rt)) > 1e-8):
        
        print("translation only")
        if (not np.all(depth == 0.5)):
        
            
            mask_p = (((mask[..., 0] / 255.0) > 0.5) * 1.0)
            depth_p, d_mask = process_depth(depth)
            # print()
            d_mean = np.mean(depth_p[(mask_p * d_mask) > 0.5])
            # d_mean = np.sum(process_depth(depth) * mask_p) / np.sum(mask_p)
            # print(d_mean)

            # Use depth map before and after projection to get the true angle
            rt[-1] += d_mean
            t = np.array([0, 0, d_mean]).astype("float32")
            # norm_t = np.linalg.norm(rt)
            # print(t, rt)
            x, y, z = cartesian_to_spherical(t[None])
            # print(rt[None].shape)
            xr, yr, zr = cartesian_to_spherical(rt[None, :])

            x_out += xr - x
            y_out += yr - y
            z_out += zr - z


            return x_out, y_out, z_out
        else:
            if rt[-1] > 0.0:
                print("scaling")
                t = np.array([0, 0, 0.5]).astype("float32")
                rt[:2] = 0.0
                # rt[-1] =
                x, y, z = cartesian_to_spherical(t[None])
                # print(rt[None].shape)
                xr, yr, zr = cartesian_to_spherical(rt[None, :])

                return xr - x, yr - y, (zr - z)


    

    return 0.0, 0.0, 0.0


def transformation_matrix_to_spherical(transform_mat):



    # # pts = np.ones((4,))
    # transform_mat = np.linalg.inv(transform_mat)
    # # T_cond = (transform_mat @ pts)[:3].astype("float32")

    # # print(T_cond.shape)

    # # print(transform_mat)
    # # transform_mat = convert_transform_geodiff_to_zero123(transform_mat)
    # R, T = transform_mat[:3, :3], transform_mat[:3, -1]
    # T_cond = -R.T @ T
    R = transform_mat[:3, :3]
    # Reflection case
    if np.linalg.det(R) < 0:
        return None, None, None

    T_cond = transform_mat[:3, -1]
    print(T_cond)
    # T_cond[0] *= -1
    # T_cond[1] *= -1
    T_cond = transform_converter[:3, :3] @ T_cond
    print("After:", T_cond)

    # transform_converter = rotateAxis(90, 2)
    # T_cond = transform_converter[:3, :3] @ T_cond

    # print("t cond: ", T_cond)

    # exit()
    # print(T_cond)

    # T_cond = transform_mat[:3, -1]
    return cartesian_to_spherical(T_cond[None, :])


def get_mask_prediction(image, h, w, model_path = "/home/ec2-user/SageMaker/test_sd/segment-anything/weights/sam_vit_h_4b8939.pth"):

    global SAM_MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if SAM_MODEL is None:
        sam = sam_model_registry["vit_h"](checkpoint=model_path).to(device)
        SAM_MODEL = sam
    else:
        sam = SAM_MODEL
    predictor = SamPredictor(sam)
    


    predictor.set_image(image)
    h = float(h)
    w = float(w)

    input_points = np.array([[image.shape[1] * w, image.shape[0] * h]])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(input_points, input_label)

    image_mask = masks[-1] * 1.0

    return image_mask


def get_initial_mask(im):

    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    rg_close = np.abs(r - g) < 10
    gb_close = np.abs(g - b) < 10
    br_close = np.abs(b - r) < 10

    all_close = np.logical_and(np.logical_and(rg_close, gb_close), br_close)


    all_close = all_close * 1.0

    return all_close

def get_point_on_obj(im):

    # im_mask = get_initial_mask(im)

    print(im.max(), im.min())
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    rg = np.abs(r.astype("float32") - 255.0)
    gb = np.abs(g.astype("float32") - 255.0)
    br = np.abs(b.astype("float32") - 255.0)
    # w = np.abs(r - 255)

    dist = rg + gb + br #+ w


    w_max = np.argmax(dist, -1)
    max_val = np.max(dist, -1)
    h_max = np.argmax(max_val)
    max_val = np.max(max_val)
    w_max = w_max[h_max]

    # print(h_max, w_max, max_val, dist[h_max, w_max])
    
    # print(np.argmax(dist, -1))
    ind = (h_max, w_max)
    # ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)

    # print(dist[ind])
    # print(ind)

    ind = (ind[0] / dist.shape[0], ind[1] / dist.shape[1])
    # exit()
    return ind
    # return all_close


def get_mask_from_output(im):

    h, w = get_point_on_obj(im)
    print(h, w)
    
    mask = get_mask_prediction(im, h, w, "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/segment-anything/weights/sam_vit_h_4b8939.pth")

    return mask


def get_mask_from_output_2(im):

    im_out = Image.fromarray(im)

    # print(im_out.size)
    mask = remove(im_out)
    mask = np.array(mask)
    mask = mask[..., -1] / 255.0
    # print(mask.shape, mask.max(), mask.min())
    # exit()
    # print(size)
    return mask

def remove_noise_from_mask(mask):

    if mask.max() <= 1.0:
        mask = (mask * 255.0).astype("uint8")

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(mask , kernel,iterations = 1)
    dilation = cv2.dilate(erosion, kernel,iterations = 1)

    mask = dilation / 255.0

    return mask


def get_mask_bounding_box(mask):


    mask = remove_noise_from_mask(mask)
    h_ind, w_ind = np.indices(mask.shape)

    if np.sum(mask) == 0:
        return 0, 1, 0, 1

    h_min, h_max = h_ind[mask > 0.5].min(), h_ind[mask > 0.5].max()
    w_min, w_max = w_ind[mask > 0.5].min(), w_ind[mask > 0.5].max()

    return h_min, h_max, w_min, w_max


def crop_image(im, mask):

    h_min, h_max, w_min, w_max = get_mask_bounding_box(mask)
    return im[h_min:h_max, w_min:w_max], mask[h_min:h_max, w_min:w_max]
    # im[mask]


def resize_image(image, aspect_ratio):

    h, w = image.shape[:2]
    ratio = aspect_ratio[1] / aspect_ratio[0]

    h, w = 512, 512

    if ratio < 1:
        new_h, new_w = h / ratio, w
    else:
        new_h, new_w = h, ratio * w

    img = cv2.resize(image, (int(new_w),int(new_h)))

    # input_img = np.array(Image.fromarray(img).resize((w, h), Image.NEAREST))
    return img


def laplacian_blend_images(im1, im2, mask):

    # Automatically figure out the size
    min_size = min(im1.shape)
    depth = int(math.floor(math.log(min_size, 2))) - 4 # at least 16x16 at the highest level.

    gauss_pyr_mask = blending.gaussPyramid(mask, depth)
    gauss_pyr_im1 = blending.gaussPyramid(im1, depth)
    gauss_pyr_im2 = blending.gaussPyramid(im2, depth)


    lapl_pyr_im1  = blending.laplPyramid(gauss_pyr_im1)
    lapl_pyr_im2 = blending.laplPyramid(gauss_pyr_im2)

    outpyr = blending.blend(lapl_pyr_im2, lapl_pyr_im1, gauss_pyr_mask)
    outimg = blending.collapse(outpyr)

    outimg = np.clip(outimg, 0, 1) * 255.0

    # outimg[outimg < 0] = 0 # blending sometimes results in slightly out of bound numbers.
    # outimg[outimg > 255] = 255
    outimg = outimg.astype(np.uint8)

    return outimg
    # return lapl_pyr_im1, lapl_pyr_im2, gauss_pyr_im1, gauss_pyr_im2, \
        # gauss_pyr_mask, outpyr, outimg
    # return


def get_blend_image_and_inpainting_mask(im, mask, im2, mask2, center, inpainted_background = None):
    h, w, _ = im.shape
    h_in, w_in, _ = im2.shape

    inpainted_background_blend = None

    


    # print(im.shape, im2.shape)
    im2_center = (h_in // 2, w_in // 2)
    # print(center, im2_center, im.shape)

    h_top = int(center[0] - im2_center[0])
    h_bottom = h_top + h_in 
    w_left = int(center[1] - im2_center[1])
    w_right = w_left + w_in


    # print(h_top, h_bottom, w_left, w_right)
    paste_im = im2
    mask_paste_im = mask2

    if h_top < 0:
        paste_im = paste_im[-h_top:]
        mask_paste_im = mask_paste_im[-h_top:]
        
        h_top = 0

    if h_bottom > h:

        d = h_bottom - h + 1
        print(d)
        paste_im = paste_im[:-d]
        mask_paste_im = mask_paste_im[:-d]
        h_bottom = h - 1

    if w_left < 0:
        paste_im = paste_im[:, -w_left:]
        mask_paste_im = mask_paste_im[:, -w_left:]
        w_left = 0
    
    if w_right > w:

        d = w_right - w + 1
        print(d)
        paste_im = paste_im[:, :-d]
        mask_paste_im = mask_paste_im[:, :-d]
        w_right = w - 1

    # print(h_top, h_bottom, w_left, w_right)

    im_obj = np.zeros_like(im)
    im_obj[h_top:h_bottom, w_left:w_right] = paste_im * mask_paste_im[..., None]



    mask_alpha = np.zeros_like(im)[..., 0]
    mask_alpha[h_top:h_bottom, w_left:w_right] = mask_paste_im
    print(im_obj.shape, im.shape)
    print("Running blending")
    im_blend = laplacian_blend_images(im / 255.0, im_obj / 255.0, mask_alpha)

    if inpainted_background is not None:
        inpainted_background_blend = cv2.resize(inpainted_background[..., :3], (int(im.shape[1]),int(im.shape[0])))

        inpainted_background_blend = laplacian_blend_images(inpainted_background_blend / 255.0, im_obj / 255.0, mask_alpha)

    # im[h_top:h_bottom, w_left:w_right] = paste_im * mask_paste_im[..., None] + (1.0 - mask_paste_im[..., None]) * im[h_top:h_bottom, w_left:w_right]
    im = im_blend
    
    # mask[h_top:h_bottom, w_left:w_right][mask_paste_im > 0.5] = 1.0
    mask = maximum_filter(mask, 25)
    mask[h_top:h_bottom, w_left:w_right][mask_paste_im > 0.5] = 0.0


    return im, mask, inpainted_background_blend


def run_and_save_zero123_single(exp_folder, model):
    os.environ["OMP_NUM_THREADS"] = "1"

    # continue
    exp_folder = complete_path(exp_folder)
    exp_dict = read_exp(exp_folder)


    im = exp_dict["input_image_png"]
    im_mask = exp_dict["input_mask_png"]
    im_out, scale_factor = get_input(im, im_mask)
    transform_mat = exp_dict["transform_npy"]
    depth = exp_dict["depth_npy"]
    transformed_mask = exp_dict["transformed_mask_square_png"]
    im_size = exp_dict["image_shape_npy"]

    ht_min, ht_max, wt_min, wt_max = get_mask_bounding_box(transformed_mask[..., 0] / 255.0)

    # print("mask: ", ht_min, ht_max, wt_min, wt_max)
    obj_center = (((ht_max + ht_min) / 2), (wt_min + wt_max)/2)


    x, y, z = transformation_matrix_to_spherical_2(transform_mat, depth, im_mask)

    print(x, y, z)
    if x == None:
        print("Skipping as det(R) < 0")
        return 
        # continue

    # exit()
    # Matching the frame of pytorch3d?

    s_time = time.perf_counter()
    im_out_gen = run_zero123(model, im_out, x=-x, y=y, z=-5*z)
    m_time = time.perf_counter()
    im_out_gen = np.array(im_out_gen[0])

    mask_out = get_mask_from_output(im_out_gen)

    if np.sum(mask_out) == 0:
        mask_out = get_mask_from_output_2(im_out_gen)


    im_obj, m_obj = crop_image(im_out_gen, mask_out)

    old_h, old_w = m_obj.shape
    new_h, new_w = old_h / scale_factor, old_w / scale_factor
    im_obj = cv2.resize(im_obj, (int(new_w),int(new_h)))
    m_obj = cv2.resize(m_obj, (int(new_w),int(new_h)))


    inpainted_background = exp_dict["inpainted_background"]
# /
    # print(inpainted_background)

    im_blended, inpainting_mask, im_blended_2 = get_blend_image_and_inpainting_mask(im, im_mask[..., 0] / 255.0, im_obj, m_obj, obj_center, inpainted_background = inpainted_background)

    e_time = time.perf_counter()

    print("Time zero123 xl (s):", m_time - s_time)
    print("remaining: ", e_time - m_time)
    print("Time (s):", e_time - s_time)

    exit()
    out_path_dir = exp_folder + "zero123/"
    os.makedirs(out_path_dir, exist_ok = True)
    

    plt.imsave(out_path_dir + "zero123_input.png", im_out)
    plt.imsave(out_path_dir + "blended.png", resize_image(im_blended, im_size))
    plt.imsave(out_path_dir + "inpainting_mask.png", resize_image(inpainting_mask, im_size), cmap="gray")
    plt.imsave(out_path_dir + "out_gen.png", im_obj)
    plt.imsave(out_path_dir + "out_gen_mask.png", m_obj, cmap="gray")

    if im_blended_2 is not None:
        plt.imsave(out_path_dir + "lama_followed_by_zero123_result.png", resize_image(im_blended_2, im_size))
        

def generate_zero_123_results(exp_root_folder, model):

    folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    folder_list.sort()


    num_k = 0

    for exp_folder in folder_list:
        
        num_k +=1
        print("Performing edit on: ", exp_folder)
        
        run_and_save_zero123_single(exp_folder, model)    
    
        # if num_k == 2:
        #     exit()

    return




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Read a directory path from the command line.")
    parser.add_argument('--path', help="Specify the directory path")
    args = parser.parse_args()

    # ckpt='105000.ckpt'
    ckpt='zero123-xl.ckpt'
    config='configs/sd-objaverse-finetune-c_concat-256.yaml'
    config = OmegaConf.load(config)
    model = initialize_models(config, ckpt)

    # exp_root_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/editing/"
    # exp_root_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/rotation_2/"
    exp_root_folder = args.path
    generate_zero_123_results(exp_root_folder, model)
    # exit()

    # model = None
    # exp_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/editing/2"
    # run_and_save_zero123_single(exp_folder, model)

    
