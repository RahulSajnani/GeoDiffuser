import os
import glob

import torch
import scipy

from test.utils import crop_and_resize, load_image, save_image, save_depth
from saicinpainting import LamaInpainter

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

import matplotlib.pyplot as plt
import numpy as np
import imageio

# Borrowed and Edited from: 
# 1. https://github.com/adobe-research/DiffusionHandles/blob/main/test/remove_foreground.py
# 2. https://github.com/adobe-research/DiffusionHandles/blob/main/test/estimate_depth.py
# 3. 
# Thank you to the authors for the code!




def count_folders(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

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


def read_txt_file(f_path):

    with open(f_path, "r") as f:
        txt = f.read()

    return txt

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
    result_ls_path = save_folder + "resized_result_ls.png"
    zero123_result_path = save_folder + "zero123/lama_followed_by_zero123_result.png"
    object_edit_result_path = save_folder + "object_edit/result_object_edit.png"
    resized_input_image = save_folder + "resized_input_image_png.png"
    resized_input_mask = save_folder + "resized_input_mask_png.png"

    prompt_path = save_folder + "prompt.txt"
    
    all_paths = [img_path, depth_path, mask_path, bg_path, depth_vis_path, transform_path, transformed_image_path, result_path, im_shape, result_ls_path, zero123_result_path, resized_input_image, object_edit_result_path, resized_input_mask, prompt_path]
    
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
            elif f_type == "txt":
                out_dict[key_name + "_txt"] = read_txt_file(f_name)
        else:
            out_dict[key_name + "_" + f_type] = None
    if out_dict["image_shape_npy"] is None:
        out_dict["image_shape_npy"] = np.array([512, 512])
    out_dict["path_name"] = d_path
    return out_dict


IMG_RES = 512


def preprocess_image(image_in, img_res=IMG_RES):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.from_numpy(image_in).unsqueeze(0).permute(0, -1, 1, 2)
    # print(image.shape)
    image = crop_and_resize(img=image, size=img_res).to(device).to(torch.float32)
    
    return image

def remove_foreground(image, fg_mask, img_res=IMG_RES, dilation=3):

    """
    Both image and fg_mask in range [0-1]
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inpainter = LamaInpainter()
    inpainter.to(device)

    image = preprocess_image(image, img_res)
    print(image.shape)

    fg_mask = preprocess_image(fg_mask, img_res)
    
    fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)


    # inpaint the foreground region to get a background image without the foreground object
    if dilation >= 0:
        fg_mask = fg_mask.cpu().numpy() > 0.5
        fg_mask = scipy.ndimage.binary_dilation(fg_mask[0, 0], iterations=dilation)[None, None, ...]
        fg_mask = torch.from_numpy(fg_mask).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        bg_img = inpainter.inpaint(image=image, mask=fg_mask)
    
    return bg_img[0]

def estimate_depth(image, img_res=IMG_RES):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = get_config("zoedepth_nk", "infer")
    depth_estimator = build_model(conf)
    depth_estimator.to(device)

    image_d = preprocess_image(image)

    with torch.no_grad():
        depth = depth_estimator.infer(image_d)

    return depth[0]
    



if __name__ == '__main__':

    exp_path = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/large_scale_study_all/large_scale_study_dataset_metrics/Translation_3D/1/"


    exp_dict = read_exp(exp_path)
    # print(exp_dict["depth_npy"].shape, exp_dict["depth_npy"].shape)
    # exit()

    # print(exp_dict.keys())

    image = exp_dict["input_image_png"] / 255.0
    mask = exp_dict["input_mask_png"][..., :1] / 255.0


    im_removed = remove_foreground(image, mask)
    print("im removed out:", im_removed.shape)
    im_removed_np = im_removed.permute(1, 2, 0).detach().cpu().numpy()
    im_removed_depth = estimate_depth(im_removed_np)
    print(im_removed_depth.shape)

    im_removed_depth_np = im_removed_depth[0].detach().cpu().numpy()


    im_removed_depth_np = im_removed_depth_np / (im_removed_depth_np.max() + 1e-6)
    imageio.imwrite("./check/im.png", exp_dict["input_image_png"])
    imageio.imwrite("./check/im_removed_depth.png", (im_removed_depth_np * 255.0).astype(np.uint8))
    save_image(im_removed, "./check/im_removed.png")
    save_depth(im_removed_depth, "./check/im_removed_depth.exr")

