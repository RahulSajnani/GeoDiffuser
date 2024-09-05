import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
from free_guidance import StableDiffusionFreeGuidancePipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn import init
from utils.guidance_functions import *
from utils.geodiffuser_reprojection import *
import argparse
from diffusers import LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from utils import *
from PIL import Image
from utils.geodiffuser_data_utils import *

os.environ["HF_HOME"] = "/oscar/scratch/rsajnani/rsajnani/research/.cache/hf"

MODE = "bilinear"


def safety_checker(clip_input, images, **kwargs):

    return images, [False]


def get_image_latent(img_path, device="cuda"):

    if img_path is None: return None
    img = Image.open(img_path)
    img = img.convert('RGB')
    image = preprocess_image(img).to(device)
    init_latents = pipe.vae.encode(image.half()).latent_dist.sample() * 0.18215

    return init_latents


def load_models(model_id = "CompVis/stable-diffusion-v1-4", device="cuda", variant="fp16"):

    pipe = StableDiffusionFreeGuidancePipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant=variant)
    pipe.unet = UNetDistributedDataParallel(pipe.unet, device_ids=[0]).cuda()

    pipe.unet = pipe.unet.module
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    torch.backends.cudnn.benchmark = True


    sd_ori_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant=variant)
    sd_ori_pipe.unet = UNetDistributedDataParallel(sd_ori_pipe.unet, device_ids=[0]).cuda()
    sd_ori_pipe.unet = sd_ori_pipe.unet.module
    sd_ori_pipe = sd_ori_pipe.to(device)
    sd_ori_pipe.scheduler = DDIMScheduler.from_config(sd_ori_pipe.scheduler.config)
    sd_ori_pipe.enable_attention_slicing()
    torch.backends.cudnn.benchmark = True

    sd_ori_pipe.safety_checker = safety_checker


    pipe_denoising_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe_noising_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)



    return pipe, sd_ori_pipe, pipe_denoising_scheduler, pipe_noising_scheduler


def run_self_guidance_folder(pipe, sd_ori_pipe, pipe_denoising_scheduler, pipe_noising_scheduler, seed, dataset_path):


    print(f"[INFO]: Running for folder: {dataset_path}")

    exp = read_exp(dataset_path)
    print(exp.keys())
    print(exp["prompt_txt"])
    prompt = exp["prompt_txt"]
    object_to_edit = prompt.split(" ")[-1]
    objects = [object_to_edit]
    print(prompt, object_to_edit, objects, exp["path_name"])
    img_path = complete_path(exp["path_name"]) + "input_image.png"



    image = exp["input_image_png"]
    mask = exp["input_mask_png"]
    depth = exp["depth_npy"]
    transform_mat = exp["transform_npy"]
    print(exp["prompt_txt"])
    K = camera_matrix(550, 550, image.shape[1] / 2.0, image.shape[0] / 2.0)

    t_w, t_h = get_geodiff_translation(image, K, depth, transform_mat, mask)
    print(t_w, t_h, exp["path_name"])


    image_torch = torch.tensor(image).float()
    image_mask = mask[..., 0]
    image_mask_torch = torch.from_numpy(image_mask).type_as(image_torch)
    depth_torch = torch.from_numpy(depth).type_as(image_torch)
    transform_mat_torch = torch.from_numpy(transform_mat).type_as(image_torch)


    t_coords_depth, p_image, projected_image_amodal_mask = get_transform_coordinates(image / 255.0, depth, image_mask / 255.0, transform_in = transform_mat_torch, return_mesh=True)

    t_coords_depth = torch.tensor(t_coords_depth)


    # image_warped = warp_grid_edit((image_torch[None].permute(0, -1, 1, 2) / 255.0).float(), t_coords_depth[None].float(), padding_mode='zeros', align_corners=True, mode=MODE)

    # p_image = image_warped.detach().cpu().numpy()[0].transpose(1, 2, 0)
    # print(p_image.max(), p_image.min())
    # plt.imsave("./projected.png", p_image)
    
    # image_warped = warp_grid_edit((torch.tensor(image_stitch[None]).permute(0, -1, 1, 2) / 255.0).float(), t_coords_depth[None].float(), padding_mode='zeros', align_corners=True, mode=MODE)
    # exit()
    



    # seed = int(torch.rand((1,)) * 100000)
    generator=torch.manual_seed(seed)

    # prompt = 'a photo of a donut and a shot of espresso on a table'
    # img_path = './img/donut.png'
    latents = get_image_latent(img_path)   
    print(latents.shape)




    # Perform Inversion

    sd_ori_pipe.scheduler = pipe_noising_scheduler
    inv_latents, _ = sd_ori_pipe(prompt=prompt, negative_prompt="", guidance_scale=guidance_scale,
                          output_type='latent', return_dict=False,
                          num_inference_steps=50, latents=latents)

    # # Reconstruct image
    # sd_ori_pipe.scheduler = pipe_denoising_scheduler
    # out = sd_ori_pipe(prompt=prompt, negative_prompt="", guidance_scale=guidance_scale,
    #                     num_inference_steps=50, latents=inv_latents)




    # Perform Edit using self guidance
    pipe.scheduler = pipe_denoising_scheduler
    
    # Move instruction to move by shape
    move = partial(perform_geometric_transform, t_coords_depth = t_coords_depth, mode = MODE )
    # guidance losses and terms
    guidance = partial(move_object_by_shape, shape_weight=1.0, appearance_weight=2.0, position_weight=8, tau=move)
    # perform the edit using diffusion self guidance
    image_list = pipe(prompt, obj_to_edit = object_to_edit, height=512, width=512, 
                    num_inference_steps=50, generator=generator, latents=inv_latents,
            max_guidance_iter_per_step=3, guidance_func=guidance, g_weight=100, guidance_scale=guidance_scale)


    im_edit = np.asarray(image_list[0].images[0])
    print(im_edit.min(), im_edit.max())

    d_path = complete_path(exp["path_name"]) + "diffusion_self_guidance/"
    os.makedirs(d_path, exist_ok=True)
    im_edit = resize_image(im_edit, exp["image_shape_npy"])

    plt.imsave(d_path + "result_diffusion_self_guidance.png", im_edit)

    # for i, image in enumerate(image_list):
    #     image.images[0].save(f"./{seed}_{i}.png")


def run_self_guidance_on_geodiffuser_data(pipe, sd_ori_pipe, pipe_denoising_scheduler, pipe_noising_scheduler, seed, exp_root_folder):

    folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    folder_list.sort()

    # print(folder_list)
    # exit()

    if check_if_exp_root(exp_root_folder):
        root_folders_list = folder_list
        for f in root_folders_list:
            folder_list = glob.glob(complete_path(f) + "**/")
            folder_list.sort()

            exp_cat = f.split("/")[-2]
            if  (exp_cat == "Removal"):
                continue

            for exp_folder in folder_list:
                run_self_guidance_folder(pipe, sd_ori_pipe, pipe_denoising_scheduler, pipe_noising_scheduler, seed, exp_folder)
    
            # exit()
    else:
        for exp_folder in folder_list:
            run_self_guidance_folder(pipe, sd_ori_pipe, pipe_denoising_scheduler, pipe_noising_scheduler, seed, exp_folder)

    pass

if __name__=="__main__":

    guidance_scale = 5.0

    torch.cuda.manual_seed_all(1234) 
    torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
    print("Start Inference!")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    pipe, sd_ori_pipe, pipe_denoising_scheduler, pipe_noising_scheduler = load_models(args.model_id)


    run_self_guidance_on_geodiffuser_data(pipe, sd_ori_pipe, pipe_denoising_scheduler, pipe_noising_scheduler, seed=args.seed, exp_root_folder = args.dataset_path)


    # run_self_guidance_folder(pipe, sd_ori_pipe, pipe_denoising_scheduler, pipe_noising_scheduler, seed=args.seed, dataset_path = args.dataset_path)









