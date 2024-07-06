import os
import glob

import torch
import scipy

# from test.utils import crop_and_resize, load_image, save_image, save_depth
from saicinpainting import LamaInpainter

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

import matplotlib.pyplot as plt
import numpy as np
import imageio
from diffhandles import DiffusionHandles

from scipy.spatial.transform import Rotation as R
import cv2

# Borrowed and Edited from: 
# 1. https://github.com/adobe-research/DiffusionHandles/blob/main/test/remove_foreground.py
# 2. https://github.com/adobe-research/DiffusionHandles/blob/main/test/estimate_depth.py
# 3. 
# Thank you to the authors for the code!


import torch
import torchvision
import imageio.v3 as imageio
import imageio.plugins as imageio_plugins

imageio_plugins.freeimage.download() # to load exr files

def load_image(path: str) -> torch.Tensor:
    # img = Image.open(path)
    # img = img.convert('RGB')
    # img = torchvision.transforms.functional.pil_to_tensor(img)

    img = torch.from_numpy(imageio.imread(path))
    if img.dim() == 2:
        img = img[..., None]
    img = img.to(dtype=torch.float32)
    img = img.permute(2, 0, 1)
    img = img / 255.0
    return img

def save_image(img: torch.Tensor, path: str):
    # img = torchvision.transforms.functional.to_pil_image(img)
    # img.save(path)

    img = img.detach().cpu()
    img = img * 255.0
    img = img.permute(1, 2, 0)
    img = img.to(dtype=torch.uint8)
    if img.shape[-1] == 1:
        img = img[..., 0]
    imageio.imwrite(path, img.numpy())
    
def load_depth(path: str) -> torch.Tensor:
    # depth = Image.open(path)
    # depth = torchvision.transforms.functional.pil_to_tensor(depth)[None,...]

    depth = torch.from_numpy(imageio.imread(path))
    if depth.dim() == 2:
        depth = depth[..., None]
    depth = depth.to(dtype=torch.float32)
    depth = depth.permute(2, 0, 1)
    return depth

def save_depth(depth: torch.Tensor, path: str):
    # depth = torchvision.transforms.functional.to_pil_image(depth, mode='F')
    # depth.save(path)

    depth = depth.detach().cpu()
    depth = depth.permute(1, 2, 0)
    depth = depth.to(dtype=torch.float32)
    depth = depth[..., 0]
    imageio.imwrite(path, depth.numpy())

def crop_and_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    if img.shape[-2] != img.shape[-1]:
        img = torchvision.transforms.functional.center_crop(img, min(img.shape[-2], img.shape[-1]))
    img = torchvision.transforms.functional.resize(img, size=(size, size), antialias=True)
    return img



DIFF_HANDLES_MODEL = None
DEPTH_MODEL = None
INPAINTING_MODEL = None

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

def remove_foreground(image, fg_mask, img_res=IMG_RES, dilation=10):

    """
    Both image and fg_mask in range [0-1]
    """
    
    global INPAINTING_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if INPAINTING_MODEL is None:
        inpainter = LamaInpainter()
        inpainter.to(device)
        INPAINTING_MODEL = inpainter
    else:
        inpainter = INPAINTING_MODEL

    image = preprocess_image(image, img_res)
    # print(image.shape)

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

    global DEPTH_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEPTH_MODEL is None:


        conf = get_config("zoedepth_nk", "infer")
        depth_estimator = build_model(conf)
        depth_estimator.to(device)
        DEPTH_MODEL = depth_estimator
    else:
        depth_estimator = DEPTH_MODEL

    image_d = preprocess_image(image)

    with torch.no_grad():
        depth = depth_estimator.infer(image_d)

    return depth[0]
    

def load_diffhandles_model(config_path=None):

    global DIFF_HANDLES_MODEL


    if DIFF_HANDLES_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diff_handles_config = OmegaConf.load(config_path) if config_path is not None else None
        diff_handles = DiffusionHandles(conf=diff_handles_config)
        diff_handles.to(device)
        DIFF_HANDLES_MODEL = diff_handles
    else:
        diff_handles = DIFF_HANDLES_MODEL

    return diff_handles

def latent_to_image(latent_image, diff_handles):

    # save image reconstructed from inversion
    with torch.no_grad():
        latent_image = 1 / 0.18215 * latent_image.detach()
        recon_image = diff_handles.diffuser.vae.decode(latent_image)['sample']
        recon_image = (recon_image + 1) / 2

    return recon_image


def convert_transform_to_diffhandles(transform_in):

    # Convert to and from pytorch3d coordinate frame
    M = np.eye(4)
    M[0, 0] = -1.0
    M[1, 1] = -1.0
    transform = np.linalg.inv(M) @ transform_in @ M

    # transform = transform_in @ M
    translation = list(transform[:3, -1])
    rotation = transform[:3, :3]
    rot_scipy = R.from_matrix(rotation)
    axis = rot_scipy.as_rotvec(degrees=True)

    angle = np.linalg.norm(axis) + 1e-8


    axis = (axis / angle)

    if np.linalg.norm(axis) < 0.1:
        axis = np.array([0.0, 1.0, 0.0])

    axis = list(axis)
    return axis, angle, translation

def resize_image(image, aspect_ratio):

    # h, w = image.shape[:2]
    ratio = aspect_ratio[1] / aspect_ratio[0]
    h, w = 512, 512

    if ratio < 1:
        new_h, new_w = h / ratio, w
    else:
        new_h, new_w = h, ratio * w

    img = cv2.resize(image, (int(new_w),int(new_h)))

    # input_img = np.array(Image.fromarray(img).resize((w, h), Image.NEAREST))
    return img

def run_geodiff_folder(exp_dict, diff_handles):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir_dh = exp_dict["path_name"] + "diffhandles/"
    os.makedirs(exp_dir_dh, exist_ok=True)
    
    image = exp_dict["input_image_png"] / 255.0
    mask = exp_dict["input_mask_png"][..., :1] / 255.0


    rot_axis, rot_angle, translation = convert_transform_to_diffhandles(exp_dict["transform_npy"])

    print(rot_angle, rot_axis, translation)

    # exit()

    im_removed = remove_foreground(image, mask)
    im_removed_np = im_removed.permute(1, 2, 0).detach().cpu().numpy()

    imageio.imwrite(exp_dir_dh + "im_removed.png", (im_removed_np * 255.0).astype(np.uint8))
    im_removed_depth = estimate_depth(im_removed_np)
    bg_depth = im_removed_depth
    fg_mask = preprocess_image(mask)

    depth = preprocess_image(exp_dict["depth_npy"][..., None])
    # Normalize depth
    depth = depth / (depth.max() + 1e-8) + 1e-2
    depth[depth > 0.95] = 1.0
    bg_depth = bg_depth / (bg_depth.max() + 1e-8) + 1e-2

    bg_depth = diff_handles.set_foreground(depth=depth, fg_mask=fg_mask, bg_depth=(bg_depth[None]))

    im_removed_depth_np = bg_depth[0, 0].detach().cpu().numpy()
    np.save(exp_dir_dh + "bg_depth_diffhandles.npy", im_removed_depth_np)
    im_removed_depth_np_norm = im_removed_depth_np / (im_removed_depth_np.max() + 1e-6)
    imageio.imwrite(exp_dir_dh + "im_removed_depth.png", (im_removed_depth_np_norm * 255.0).astype(np.uint8))

    prompt = "a fox"
    img = preprocess_image(image)

    null_text_emb, init_noise = diff_handles.invert_input_image(img, depth, prompt)
    null_text_emb, init_noise, activations, latent_image = diff_handles.generate_input_image(
                depth=depth, prompt=prompt, null_text_emb=null_text_emb, init_noise=init_noise)


    recon_image = latent_to_image(latent_image, diff_handles)
    save_image(recon_image.clamp(min=0, max=1)[0], exp_dir_dh + "recon.png")
    # print("Saved Reconstruction Image for Check")



    # get transformation parameters
    translation = torch.tensor(translation, dtype=torch.float32)
    rot_axis = torch.tensor(rot_axis, dtype=torch.float32) 
    rot_angle = float(rot_angle) 

    results = diff_handles.transform_foreground(
    depth=depth, prompt=prompt,
    fg_mask=fg_mask, bg_depth=bg_depth,
    null_text_emb=null_text_emb, init_noise=init_noise,
    activations=activations,
    rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
    use_input_depth_normalization=False)

    if diff_handles.conf.guided_diffuser.save_denoising_steps:
        edited_img, edited_disparity, denoising_steps = results
    else:
        edited_img, edited_disparity = results
        denoising_steps = None

    save_image((edited_disparity/edited_disparity.max())[0], exp_dir_dh + "im_disparity_transformed.png")
    save_image(edited_img[0], exp_dir_dh + "im_edited_diffhandles_square.png")


    resized_edit_image = resize_image(edited_img[0].permute(1, 2, 0).detach().cpu().numpy(), exp_dict["image_shape_npy"])


    plt.imsave(exp_dir_dh + "im_edited_diffhandles.png", resized_edit_image)
    print("[INFO]: Saving Edited Image to location: ", exp_dir_dh + "im_edited_diffhandles.png")
    



if __name__ == '__main__':

    print("Running Script!!!!!!!!!!!!!!")

    
    diff_handles = load_diffhandles_model()

    exp_path = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/large_scale_study_all/large_scale_study_dataset_metrics/Rotation_3D/14/"

    exp_dict = read_exp(exp_path)
    run_geodiff_folder(exp_dict, diff_handles)

    exit()

    