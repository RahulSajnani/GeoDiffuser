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
    
    all_paths = [img_path, depth_path, mask_path, bg_path, depth_vis_path, transform_path, transformed_image_path, result_path, im_shape]
    
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

    image = PIL.Image.fromarray(np.array(im_out))
    
    # resize image such that long edge is 512
    image.thumbnail([200, 200], PIL.Image.Resampling.LANCZOS)
    image = add_margin(image, (255, 255, 255), size=256)
    image = np.array(image)

    image = (image / 255.0).astype(np.float32)

    return image

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


def transformation_matrix_to_spherical_2(transform_mat):

    r = transform_mat[:3, :3]
    rt = transform_mat[:3, -1]
    # print(r)
    dist = np.sum(np.abs(r - np.eye(3))) 
    # print(dist)
    if dist > 1e-8:
        t = np.array([0, 0, 1]).astype("float32")
        x, y, z = cartesian_to_spherical(t[None])
        rt = r.T @ t
        xr, yr, zr = cartesian_to_spherical(rt[None, :])
        return xr - x, yr - y, zr - z
    elif np.sum(np.abs(rt)) > 1e-8:
        rt[-1] += 1
        norm_t = np.linalg.norm(rt)
        t = np.array([0, 0, 1]).astype("float32") * norm_t
        # print(t, rt)
        x, y, z = cartesian_to_spherical(t[None])
        # print(rt[None].shape)
        xr, yr, zr = cartesian_to_spherical(rt[None, :])
        return xr - x, yr - y, zr - z

    else:
        return 0, 0, 0


def transformation_matrix_to_spherical(transform_mat):



    # # pts = np.ones((4,))
    # transform_mat = np.linalg.inv(transform_mat)
    # # T_cond = (transform_mat @ pts)[:3].astype("float32")

    # # print(T_cond.shape)

    # # print(transform_mat)
    # # transform_mat = convert_transform_geodiff_to_zero123(transform_mat)
    # R, T = transform_mat[:3, :3], transform_mat[:3, -1]
    # T_cond = -R.T @ T
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




if __name__ == "__main__":


    # t = rotateAxis(90, 0) @ translateMatrix(0, 0, 1)
    # print(transformation_matrix_to_spherical(t))

    # t = rotateAxis(90, 1) @ translateMatrix(0, 0, 1)
    # print(t)
    # print(transformation_matrix_to_spherical(t))

    # t = rotateAxis(90, 2) @ translateMatrix(0, 0, 1)
    # # print(t)
    # print(transformation_matrix_to_spherical(t))

    # exit()

    exp_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/editing/64"

    exp_dict = read_exp(exp_folder)

    # list_exp_details(exp_dict)


    

    im = exp_dict["input_image_png"]
    im_mask = exp_dict["input_mask_png"]
    im_out = get_input(im, im_mask)
    transform_mat = exp_dict["transform_npy"]
    # print(transform_mat)
    # print(transform_mat)
    x, y, z = transformation_matrix_to_spherical_2(transform_mat)
    # x_i, y_i, z_i = transformation_matrix_to_spherical(np.eye(4).astype("float32"))

    # print(x - x_i, y - y_i, z - z_i)
    # # y should be 19 for this
    # print("current: ", x, y, z)
    # x = 0
    # y = 19
    # z = 0
    # print("expected:", x, y, z)
    # x = x - 90
    # y = y - 90
    # z = z - 0
    print("theta = ", x, "phi = ", y, z)
    # exit()

    ckpt='105000.ckpt'
    config='configs/sd-objaverse-finetune-c_concat-256.yaml'
    config = OmegaConf.load(config)
    model = initialize_models(config, ckpt)

    plt.imsave("./out_in.png", im_out)
    im_out_gen = run_zero123(model, im_out, x=x, y=y, z=z)

    im_out_gen = im_out_gen[0]
    # print(im_out_gen.shape, im_out_gen.min(), im_out_gen.max())
    plt.imsave("./out_gen.png", np.array(im_out_gen))




    pass
