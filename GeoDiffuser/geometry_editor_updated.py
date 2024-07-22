import os
# os.environ['HF_HOME'] = "/home/ec2-user/SageMaker/.cache/hf"
os.environ['HF_HOME'] = "/oscar/scratch/rsajnani/rsajnani/research/.cache/hf"
import diffusers
# diffusers.__version__
# !nvidia-smi
import xformers.ops as xops
import logging

from diffusers.models.attention_processor import USE_PEFT_BACKEND
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
import gc
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import shutil
from torch.optim.adam import Adam
from PIL import Image
from util import softsplat
from util.io import log_args
from util.warp_utils import RasterizePointsXYsBlending, GaussianSmoothing
from skimage.exposure import match_histograms
from util.image_processing import masked_histogram_matching
import random
import pytorch_lightning as pl

import torch.nn.functional as F
import torchvision.transforms as T

torch.cuda.is_available()

MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 4.0
MAX_NUM_WORDS = 77
IMAGE_SIZE = 512
SKIP_OPTIM_STEPS = 0
SEED = 1234
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MODE="bilinear"
SPLATTER = RasterizePointsXYsBlending()
# USE_PEFT_BACKEND = False
# UNCOND_TEXT="pixelated, unclear, blurry, grainy"
UNCOND_TEXT=""
# DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
DIFFUSION_MODEL = "CompVis/stable-diffusion-v1-4"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-1-base"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-base"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

LDM_STABLE = None
SCHEDULER = None
TOKENIZER = None
UNET_NAME = None
PROGRESS_BAR = None



from torch.profiler import profile, record_function, ProfilerActivity
import cv2
# MODE="nearest"



GAUSSIAN_FEATURE_SMOOTHER = GaussianSmoothing(dim=2, kernel_size=5)
def torch_erode(A, kernel=3):

    k = torch.ones(1, 1, kernel, kernel).type_as(A)

    k_sum = torch.sum(k)
    conv_out = torch.nn.functional.conv2d(A, k, padding=kernel // 2)

    conv_mask = (conv_out == k_sum) * 1.0
    out_f = torch.ones_like(A) * conv_mask
    # print(torch.sum(out_f), "sum")

    return out_f

def torch_dilate(A, kernel=3):

    k = torch.ones(1, 1, kernel, kernel).type_as(A)

    k_sum = 1
    conv_out = torch.nn.functional.conv2d(A, k, padding=kernel // 2)

    conv_mask = (conv_out >= k_sum) * 1.0
    out_f = torch.ones_like(A) * conv_mask
    # print(torch.sum(out_f), "sum")


    return out_f




def norm_tensor(A, eps = 1e-12):

    return torch.sqrt(torch.sum(A * A) + eps)


def adaptive_optimization_step_editing(controller, i, skip_optim_steps, out_loss_log_dict):



    if (i / NUM_DDIM_STEPS) < 0.4:
        remaining_steps = int((0.4 - (i / NUM_DDIM_STEPS)) * NUM_DDIM_STEPS / skip_optim_steps)
        expected_removal_loss_value = -1.5 / (1.25)**(remaining_steps)
        expected_movement_loss_value = 0.4 / (1.1)**(remaining_steps)

        expected_sim_loss_value = 0.5 / (1.05)**(remaining_steps)

        print(i, " remaining_steps: ", remaining_steps, " expected removal loss: ", expected_removal_loss_value, " current loss: ", out_loss_log_dict["self"]["removal"])
        # print(i, " remaining_steps: ", remaining_steps, " expected movement loss: ", expected_movement_loss_value, " current loss: ", out_loss_log_dict["self"]["movement"])
        # print(i, " remaining_steps: ", remaining_steps, " expected movement loss: ", expected_sim_loss_value, " current loss: ", out_loss_log_dict["self"]["sim"])

        # if (expected_movement_loss_value < out_loss_log_dict["self"]["movement"]):
        #     print("increasing movement loss weight")
        #     controller.loss_weight_dict["self"]["movement"] *= 1.5
        
        # if (expected_sim_loss_value < out_loss_log_dict["self"]["sim"]):
        #     print("increasing sim loss weight")
            # controller.loss_weight_dict["self"]["sim"] *= 1.05
        # elif (expected_movement_loss_value > out_loss_log_dict["self"]["movement"]):
        #     controller.loss_weight_dict["self"]["movement"] *= 0.5



        if (expected_removal_loss_value < out_loss_log_dict["self"]["removal"]):
            # print(controller.loss_weight_dict)
            print("increasing removal loss weight")
            controller.loss_weight_dict["self"]["removal"] *= 1.3
            # print(controller.loss_weight_dict)

        elif (2.5 * expected_removal_loss_value > out_loss_log_dict["self"]["removal"]):
            # print(controller.loss_weight_dict)
            print("reducing removal loss weight")
            controller.loss_weight_dict["self"]["removal"] /= 2.0
            # print(controller.loss_weight_dict)
        
    elif ((i / NUM_DDIM_STEPS) > 0.4) and ((i / NUM_DDIM_STEPS) < 0.8):

        if (-1.8 < out_loss_log_dict["self"]["removal"]):
            controller.loss_weight_dict["self"]["removal"] *= 2.0
        else:
            controller.initialize_default_loss_weights()

    else:
        controller.initialize_default_loss_weights()
        # print("reinitialized controller loss weights: ", controller.loss_weight_dict)

def adaptive_optimization_step_remover(controller, i, skip_optim_steps, out_loss_log_dict):



    if (i / NUM_DDIM_STEPS) < 0.4:
        remaining_steps = int((0.4 - (i / NUM_DDIM_STEPS)) * NUM_DDIM_STEPS / skip_optim_steps)
        expected_removal_loss_value = -1.5 / (1.25)**(remaining_steps)
        expected_movement_loss_value = 0.4 / (1.1)**(remaining_steps)

        expected_sim_loss_value = 0.5 / (1.05)**(remaining_steps)

        print(i, " remaining_steps: ", remaining_steps, " expected removal loss: ", expected_removal_loss_value, " current loss: ", out_loss_log_dict["self"]["removal"])
        
        # if (expected_movement_loss_value < out_loss_log_dict["self"]["movement"]):
        #     print("increasing movement loss weight")
        #     controller.loss_weight_dict["self"]["movement"] *= 1.5
        
        # if (expected_sim_loss_value < out_loss_log_dict["self"]["sim"]):
        #     print("increasing sim loss weight")
            # controller.loss_weight_dict["self"]["sim"] *= 1.05
        # elif (expected_movement_loss_value > out_loss_log_dict["self"]["movement"]):
        #     controller.loss_weight_dict["self"]["movement"] *= 0.5



        if (expected_removal_loss_value < out_loss_log_dict["self"]["removal"]):
            # print(controller.loss_weight_dict)
            print("increasing removal loss weight")
            controller.loss_weight_dict["self"]["removal"] *= 1.3
            # print(controller.loss_weight_dict)

        elif (2.5 * expected_removal_loss_value > out_loss_log_dict["self"]["removal"]):
            # print(controller.loss_weight_dict)
            print("reducing removal loss weight")
            controller.loss_weight_dict["self"]["removal"] /= 2.5
            # print(controller.loss_weight_dict)
        
    elif ((i / NUM_DDIM_STEPS) > 0.4) and ((i / NUM_DDIM_STEPS) < 0.8):

        if (-1.8 < out_loss_log_dict["self"]["removal"]):
            controller.loss_weight_dict["self"]["removal"] *= 2.0
        else:
            controller.initialize_default_loss_weights()

    else:
        controller.initialize_default_loss_weights()
        # print("reinitialized controller loss weights: ", controller.loss_weight_dict)



def adaptive_optimization_step_stitching(controller, i, skip_optim_steps, out_loss_log_dict):


    # if (i / NUM_DDIM_STEPS) > 0.4 and ((i / NUM_DDIM_STEPS) < 0.45):
    #     controller.loss_weight_dict["self"]["sim_out"] /= 1.5
    #     controller.loss_weight_dict["cross"]["sim_out"] /= 1.5
    #     print("Reducing sim loss")
    
    # return
    if (i / NUM_DDIM_STEPS) < 0.4:
        remaining_steps = int((0.4 - (i / NUM_DDIM_STEPS)) * NUM_DDIM_STEPS / skip_optim_steps)
        expected_sim_loss_value = 0.18 / (1.01)**(remaining_steps)
        # expected_movement_loss_value = 0.4 / (1.1)**(remaining_steps)

        # expected_sim_loss_value = 0.5 / (1.05)**(remaining_steps)

        # print(i, " remaining_steps: ", remaining_steps, " expected removal loss: ", expected_removal_loss_value, " current loss: ", out_loss_log_dict["self"]["removal"])
        # print(i, " remaining_steps: ", remaining_steps, " expected movement loss: ", expected_movement_loss_value, " current loss: ", out_loss_log_dict["self"]["movement"])
        print(i, " remaining_steps: ", remaining_steps, " expected movement loss: ", expected_sim_loss_value, " current loss: ", out_loss_log_dict["self"]["sim_out"])

        # if (expected_movement_loss_value < out_loss_log_dict["self"]["movement"]):
        #     print("increasing movement loss weight")
        #     controller.loss_weight_dict["self"]["movement"] *= 1.5
        
        # if (expected_sim_loss_value < out_loss_log_dict["self"]["sim"]):
        #     print("increasing sim loss weight")
            # controller.loss_weight_dict["self"]["sim"] *= 1.05
        # elif (expected_movement_loss_value > out_loss_log_dict["self"]["movement"]):
        #     controller.loss_weight_dict["self"]["movement"] *= 0.5



        if (expected_sim_loss_value < out_loss_log_dict["self"]["sim_out"]):
            # print(controller.loss_weight_dict)
            print("increasing sim loss weight")
            controller.loss_weight_dict["self"]["sim_out"] *= 1.1
            # print(controller.loss_weight_dict)

        elif (2.5 * expected_sim_loss_value > out_loss_log_dict["self"]["sim_out"]):
            # print(controller.loss_weight_dict)
            print("reducing sim loss weight")
            controller.loss_weight_dict["self"]["sim_out"] /= 2.5
            # print(controller.loss_weight_dict)
        
    elif ((i / NUM_DDIM_STEPS) > 0.4) and ((i / NUM_DDIM_STEPS) < 0.7):

        if (0.2 < out_loss_log_dict["self"]["sim_out"]):
            controller.loss_weight_dict["self"]["sim_out"] *= 1.1
        else:
            controller.initialize_default_loss_weights()

    else:
        controller.initialize_default_loss_weights()
        # print("reinitialized controller loss weights: ", controller.loss_weight_dict)



def update_loss_log_dict(loss_log_dict_1, loss_log_dict_2):

    for key in loss_log_dict_1:
        loss_log_dict_1[key] += loss_log_dict_2[key]

    return loss_log_dict_1


def clear_controller_loss(controller):

    del controller.loss 
    controller.loss = 0.0

    if controller.loss_log_dict is not None:
        controller.initialize_loss_log_dict()


def convert_loss_log_to_numpy(loss_log_dict):

    out_dict = {"self": {}, "cross": {}}
    for att_type in loss_log_dict:
        if att_type == "self" or att_type == "cross":
            for key in loss_log_dict[att_type]:
                out_dict[att_type][key] = loss_log_dict[att_type][key].item()
        else:
            out_dict[att_type] = loss_log_dict[att_type]

    return out_dict

def cosine_sim(a, b, dim_l = -1):
    # a - b, f, h*w, d
    # b - b, f, h*w, d
    # sim - b, f, h*w

    a_n = torch.nn.functional.normalize(a, dim=dim_l)
    b_n = torch.nn.functional.normalize(b, dim=dim_l)
    

    sim = torch.sum(a_n * b_n, dim_l)

    return sim

def get_correlation_loss_stitch(correlation, mask_zero, mask_one, mask_warped_one):


    # print(correlation.shape, mask_zero.shape, mask_one.shape, mask_warped_one.shape)
    f = correlation.shape[0]

    correlation_inpaint = correlation * mask_zero[..., 0]
    correlation_wo_edit = correlation * mask_one[..., 0]

    
    p_correlation_inpaint = torch.max(correlation_inpaint, -1).values * mask_warped_one[0, :1, :, 0]
    p_correlation_wo_edit = torch.max(correlation_wo_edit, -1).values * mask_warped_one[0, :1, :, 0]

    loss_correlation = torch.sum(-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4)) / (torch.sum(mask_warped_one) * f + 1e-8)

    return loss_correlation

def gradient(r_out):

    D_dw = r_out[:, :, :, 1:] - r_out[:, :, :, :-1]
    D_dh = r_out[:, :, 1:, :] - r_out[:, :, :-1, :]
    return D_dh, D_dw


def get_smoothness_loss(replace_out):

    b, f, hw, d = replace_out.shape

    h = int(np.sqrt(hw))
    w = h
    r_out = replace_out.reshape(b, f, h, w, d)

    D_dh, D_dw = gradient(r_out)

    loss = D_dh.abs().mean() + D_dw.abs().mean()

    return loss, D_dh.abs(), D_dw.abs()



def setup_color_correction(image):
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2HSV)
    return correction_target


def apply_color_correction(correction, image):
    # Match hue and saturation

    image_hsv_corrected = match_histograms(
        setup_color_correction(image),
        setup_color_correction(correction),
        channel_axis=-1)
    
    image_in = setup_color_correction(image)
    image_in[..., :2] = image_hsv_corrected[..., :2]
    image = cv2.cvtColor(np.asarray(image_in.copy()), cv2.COLOR_HSV2RGB).astype("uint8")

        


    

    return image


# @torch.no_grad()
def warp_grid_edit(src, t_coords, padding_mode = None, mode = None, align_corners = False, depth=None, use_softsplat = True, splatting_radius = None, splatting_tau = None, splatting_points_per_pixel = None):

    if use_softsplat:

        if splatting_radius is not None:
            SPLATTER.radius = splatting_radius
        if splatting_tau is not None:
            SPLATTER.tau = splatting_tau
        if splatting_points_per_pixel is not None:
            SPLATTER.points_per_pixel = splatting_points_per_pixel
            
        store_device = str(src.device)
        if not str(src.device).startswith("cuda"):
            src = src.to("cuda")
            t_coords = t_coords.to("cuda")


        b, f, h, w = src.shape



        t_coords_in = t_coords.reshape(b, h*w, -1)#.float32()

        src_in = src.reshape(b, f, h*w)#.float32()
        out = SPLATTER(t_coords_in, src_in)

        if not store_device.startswith("cuda"):
            out = out.to(store_device)
    else:
        padd_mode = "zeros"
        m_mode = MODE
        if padding_mode is not None:
            padd_mode = padding_mode
        
        if mode is not None:
            m_mode = mode

        out = F.grid_sample(src, t_coords, padding_mode=padd_mode, align_corners=align_corners, mode=m_mode)

    return out

class VanillaAttentionProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self,):
        pass

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(None, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def load_model(unet_path = ""):

    global UNET_NAME

    data_type = torch.half
    # ldm_stable = StableDiffusionXLPipeline.from_pretrained(DIFFUSION_MODEL, use_auth_token=MY_TOKEN, torch_dtype=torch.half).to(DEVICE)
    ldm_stable = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL, use_auth_token=MY_TOKEN, torch_dtype=torch.half).to(DEVICE)
    # print(ldm_stable.scheduler.config)
    # scheduler = DDIMScheduler.from_pretrained(DIFFUSION_MODEL, subfolder='scheduler', beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    # variant="fp16"
    ldm_stable.scheduler = scheduler
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # ldm_stable = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL, use_auth_token=MY_TOKEN, scheduler=scheduler).to(DEVICE)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer
    
    ldm_stable.unet.set_attn_processor(VanillaAttentionProcessor())
    
    if DIFFUSION_MODEL.split("-")[-2] == "v1":
        print("[INFO]: Using Updated vae")
        ldm_stable.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.half).to(DEVICE).eval()

    # ldm_stable.vae.set_attn_processor(diffusers.models.attention_processor.AttnProcessor())

    UNET_NAME = DIFFUSION_MODEL
    if unet_path != "":
        print("[INFO]: Loading UNET model from path: ", unet_path)
        ldm_stable.unet = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder="unet", torch_dtype=data_type).to(DEVICE)
        UNET_NAME = unet_path
    
    # ldm_stable.unet = torch.compile(ldm_stable.unet, mode = "reduce-overhead")
    ldm_stable.unet = ldm_stable.unet.eval()

    



    # print("compile done!")
    # print(ldm_stable.unet.config)
    # exit()
    
    return ldm_stable, tokenizer, scheduler


# ptp

def transform_mask(mask_in, t_mask):
    
    mask = mask_in[0:1].detach()#.clone()
    mask = mask[:, None]
    mask = mask.tile((t_mask.shape[0], t_mask.shape[1], 1, 1)).type_as(t_mask)
    s_z = t_mask.shape[-1]
    mask = T.Resize(size=(s_z, s_z), antialias=False)(mask)
    #.tile((t_mask.shape))
    return mask

@torch.no_grad()
def get_3d_transform_coords(attn_layer, t_coords_in):
    
    # print("In get 3d transform coords")
    # print(attn_layer.shape)
    
    b, c, d, h, w = attn_layer.shape
    new_l = torch.linspace(0, attn_layer.shape[2], steps=attn_layer.shape[2])
    denom = ((attn_layer.shape[2] / 2.0))
    if denom == 0:
        denom = 1e-8
        
    new_l = (new_l - (attn_layer.shape[2] / 2.0)) / denom
    new_l = new_l[None, :, None, None, None].tile((b, 1, h, w, 1)).type_as(t_coords_in)
    # assert False, "breakpoint"
    t_coords = t_coords_in[:, None].tile((1, attn_layer.shape[2], 1, 1, 1))
    t_coords = torch.cat([new_l, t_coords], -1)
    # print(t_coords.shape, attn_layer.shape)
    # assert False, "breakpoint"
    return t_coords

def smooth_mask(mask, k = 1):
    
    # print(mask.shape, " old")
    # if mask.shape[-1] <= 32:
    #     k = 1
    # elif mask.shape[-1] <= 64:
    #     k = 1
    # else:
    #     k = 2
    
    
    mask = nnf.max_pool2d(mask, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
    # print(mask.shape, " mid")
    # mask = nnf.interpolate(mask, size=(mask.shape[2:]))
    # print(mask.shape, " f")
    return mask

from torchvision.transforms import v2

@torch.no_grad()
def interpolate_from_mask(features, foreground_mask, distance):

    ''' 
    features - b, h, n^2, D
    mask - 1, 1, n^2, 1
    distance - b/1, n^2, n^2

    '''
    # print(foreground_mask.shape)

    # Setting background distances to very high value
    distance_new = distance * 512 / 2.0 + 100000 * (1.0 - (foreground_mask[:1, :1, :, 0] > 0.5) * 1.0)
    inv_distance = 1.0 / (distance_new + 1e-4)
    max_4_inv_distances = torch.topk(inv_distance, k=4, dim=-1, largest=True, sorted=False)



    m_indices = (max_4_inv_distances.indices[:, None, :] * torch.ones_like(features[..., 0])[..., None]).long() # b, h, n^2, 4
    m_values = max_4_inv_distances.values[:, None, :] * torch.ones_like(features[..., 0])[..., None]

    # print(m_indices.shape, features.shape, inv_distance.shape)
    b_idx = (torch.arange(0, features.shape[0])[:, None, None, None].type_as(m_indices) * torch.ones_like(m_indices)).type_as(m_indices)

    m_indices = torch.clip(m_indices, min=0, max = features.shape[2] - 1)
    h_idx = torch.arange(0, features.shape[1])[None, :, None, None].type_as(m_indices)

    h_idx = (h_idx * torch.ones_like(m_indices)).type_as(m_indices)

    features_select = features[b_idx.long(), h_idx.long(), m_indices] # b, h, n^2, 4, D
    feature_inv_distances=m_values # b, h, n^2, 4

    interpolated_features = torch.sum(features_select * feature_inv_distances[..., None], -2) / (torch.sum(feature_inv_distances, -1)[..., None] + 1e-12) # b, h, n^2, D

    distance_weights = torch.exp(- (1 / torch.max(feature_inv_distances, -1).values) / 5) # b, h, n^2

    return interpolated_features, distance_weights



def smooth_mask_gaussian(mask, k = 1):
    
    blurrer = v2.GaussianBlur(kernel_size=(2*k+1, 2*k+1), sigma=(2*k / 6))
    # print(mask.shape, " old")
    # if mask.shape[-1] <= 32:
    #     k = 1
    # elif mask.shape[-1] <= 64:
    #     k = 1
    # else:
    #     k = 2

    mask = blurrer(mask)
    

    
    # mask = nnf.max_pool2d(mask, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
    # print(mask.shape, " mid")
    # mask = nnf.interpolate(mask, size=(mask.shape[2:]))
    # print(mask.shape, " f")
    return mask

def regularize_mask(mask):
    
    # print(mask.shape)
    reg_mask = torch.mean(mask, 1, keepdims=True).tile((1, mask.shape[1], 1))
    # print(reg_mask.shape)
    
    return reg_mask


def fix_dir_path(d_path):
    return os.path.join(d_path, "")
    
    


def baddbmm_bmm(a, b, scale = 1):

    b_in = torch.empty(
                a.shape[0], a.shape[1], b.shape[-1], dtype=a.dtype, device=a.device
            )
    sim = torch.baddbmm(b_in, a, b, beta=0, alpha = scale)

    return sim

# @torch.jit.script
def compute_attention(q, k, scale, mask = None, fg_mask_warp = None, fg_mask = None, inpaint_mask = None):

    # sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale

    # print(q.shape, k.shape)
    b_in = torch.empty(
                q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device
            )
    sim = torch.baddbmm(b_in, q, k.permute(0, -1, 1), beta=0, alpha = scale) #* scale
    
    # if mask is not None:
    #     mask = mask.reshape(batch_size, -1)
    #     max_neg_value = -torch.finfo(sim.dtype).max
    #     mask = mask[:, None, :].repeat(h, 1, 1)
    #     sim.masked_fill_(~mask, max_neg_value)

    if fg_mask_warp is not None:
        # print(sim[:, fg_mask_warp[0, 0].reshape(-1) >= 0.5, :].shape, fg_mask_warp.shape, fg_mask.shape)
        # print(sim.min(), sim.max(), " sim min and max")

        # Ensure that the forward warped mask is attending to the object of interest!
        sim[:, fg_mask_warp[0, 0].reshape(-1) >= 0.5, :][..., fg_mask[0, 0].reshape(-1) < 0.5] = -1000.0

        # Ensure the inpainting mask does not attend to foreground
        sim[:, inpaint_mask[0, 0].reshape(-1) >= 0.5, :][..., fg_mask[0, 0].reshape(-1) >= 0.5] = -1000.0


        # exit()
    # attention, what we cannot get enough of
    attn = F.softmax(sim, dim=-1)
    
    return attn
        
        


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t, transform_coords):
        # print("step callback")
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale = None, mask = None):
        raise NotImplementedError

    def __call__(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale=None, mask = None):
       
        if self.cur_att_layer >= self.num_uncond_att_layers:
            
            if (type(self).__name__.startswith("AttentionGeometry")):
                if LOW_RESOURCE:
                    out = self.forward(q, k, v, is_cross, place_in_unet, transform_coord=transform_coords, scale = scale, mask = mask)
                else:
                    # attn = compute_attention(q, k, scale, mask)
                    # # print(q.shape, "highest")
                    # out_1 = torch.einsum("b i j, b j d -> b i d", attn, v)
                    # h = attn.shape[0]
                    # out_edit = self.forward(q[h // 2:], k[h // 2:], v[h // 2:], is_cross, place_in_unet, transform_coords = transform_coords, scale = scale, mask = mask)
                    # print(out_edit.shape, out_1.shape)
                    out_edit = self.forward(q, k, v, is_cross, place_in_unet, transform_coords = transform_coords, scale = scale, mask = mask)
                    out = out_edit
                    # print("before: ", out_1.shape,  " after :", out.shape)
                    # out = torch.cat([out_1[:h//2], out_edit], 0)
            else:
                out = self.forward(q, k, v, is_cross, place_in_unet, transform_coords=transform_coords, scale = scale, mask = mask)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return out
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0




class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords=None, scale = None, mask = None):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        attn = compute_attention(q, k, scale, mask)
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn.detach())
            
        out = torch.bmm(attn, v)
        return out

    
    def attn_store(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn.detach())
            # print("storing attention of shape: ", attn.shape, " ", key, " length: ", len(self.step_store[key]))
        return 

    def between_steps(self):
        # print("Running between step")
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.step_store:
                self.attention_store[key] = self.attention_store[key] + self.step_store[key]

                if self.cur_step == 1:
                    self.attention_store["length_" + key] = len(self.step_store[key])

                # for i in range(len(self.step_store[key])):
                #     if len(self.attention_store[key]) > 0:
                #         self.attention_store[key][i] += self.step_store[key][i]
                #     else:
                #         self.attention_store[key].append(self.step_store[key][i]) 

        # print(self.attention_store, self.step_store)
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
        
        
def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()




def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


def smooth_and_reshape_attention_mask(mask, in_mat = None, in_mat_shape = None):
    

    s_m = int((mask.shape[-1]))
    
    if in_mat is not None:
        s_z = in_mat.shape[-1]
    else:
        s_z = in_mat_shape[-1]
        
    m = torch.reshape(mask, (mask.shape[0], mask.shape[1], s_m, s_m))

    resize_factor = int(np.log2(s_m / s_z))

    # print(s_m, s_z, resize_factor, s_z * 2**resize_factor)

    # print(m.shape, "initial", in_mat_shape)
    assert s_z * 2**resize_factor == s_m, "size mismatch"

    for _ in range(resize_factor):

        m = smooth_mask_gaussian(m)
        m = T.Resize(size=(m.shape[-2] // 2, m.shape[-1] // 2))(m)


    # print(m.shape, "final")
        
    return m

# NT

from torchvision import transforms

STATE = torch.get_rng_state()

def set_seed_apply_transform(im, state = None, fill = 0.0, affine = 1):
    #     https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image
    if state is not None:
        torch.set_rng_state(state)
    
    state = torch.get_rng_state()
    
    if affine:
        im_out = T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.3), scale=(0.8, 0.95), fill = fill)(im)
    else:
        im_out = T.RandomPerspective(distortion_scale=0.6, p=1.0, fill = fill)(im)
    
    
    return im_out, state

def load_256(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape

    image = np.array(Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE)))
    return image


class NullInversion:
    def null_text_optimization_transforms_loss(self, t, latent_cur, cond_embeddings, uncond_embeddings, latent_prev, noise_pred_i, t_coords=None):
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        
        
        
        if t_coords is None:
            latent_cur_t, state = set_seed_apply_transform(latent_cur, fill = 0.0, affine = 1)
            latent_cur_t_1, state = set_seed_apply_transform(latent_cur, fill = 1.0, affine = 1)
            latent_prev_t, _ = set_seed_apply_transform(latent_prev, fill = 0.0, affine = 1)
        else:
        #             print(t_coords.shape, latent_cur.shape)
            s = latent_cur.shape[-1]
            t_coords_new = t_coords.permute(0, -1, 1, 2)
            t_coords_new = T.Resize(size=(s, s))(t_coords_new).permute(0, 2, 3, 1).type_as(latent_cur)
            latent_cur_t = warp_grid_edit(latent_cur, t_coords_new, padding_mode='zeros', align_corners=True, mode="bilinear")
            latent_cur_t_1 = warp_grid_edit(latent_cur, t_coords_new, padding_mode='reflection', align_corners=True, mode="bilinear")
            latent_prev_t = warp_grid_edit(latent_prev, t_coords_new, padding_mode='reflection', align_corners=True, mode="bilinear")
            #             assert False, "break"
        #             latent_cur_t = warp_grid_edit
        #         Get mask for content 
        mask_content = (torch.abs(latent_cur_t_1 - latent_cur_t) > 0.1) * 1.0
        latent_cur_t = latent_cur_t * mask_content + (1.0 - mask_content) * noise_pred_i * (1 - alpha_prod_t) ** 0.5
        with torch.no_grad():
            noise_pred_cond = self.get_noise_pred_single(latent_cur_t, t, cond_embeddings)

        noise_pred_uncond = self.get_noise_pred_single(latent_cur_t, t, uncond_embeddings)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
        latents_prev_rec_t = self.prev_step(noise_pred, t, latent_cur_t)
        latent_prev_t = latent_prev_t * mask_content + (1.0 - mask_content) * noise_pred * (1 - alpha_prod_t) ** 0.5
        
        loss = nnf.mse_loss(latents_prev_rec_t, latent_prev_t)
        # print(controller.)
        # aggregate_attention(self.controller)
        return loss 
    
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        # torch.set_rng_state(STATE)
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        # torch.set_rng_state(STATE)
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        # torch.set_rng_state(STATE)
        latents = latents.detach() / self.model.vae.config.scaling_factor
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * self.model.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [UNCOND_TEXT], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, latent_2 = None):

        self.model.unet.set_attn_processor(VanillaAttentionProcessor())
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if latent_2 is not None:
            latent = torch.cat([latent, latent_2], 0)

        all_latent = [latent]
        all_noise = [latent]
        latent = latent.clone().detach()

        inverse_scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

        # inverse_scheduler = DDIMInverseScheduler.from_pretrained(DIFFUSION_MODEL, subfolder='scheduler', beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=0)#, timestep_spacing = "leading")


        # timesteps, num_inference_steps = self.model.retrieve_timesteps(inverse_scheduler, NUM_DDIM_STEPS, DEVICE, None)

        inverse_scheduler.set_timesteps(NUM_DDIM_STEPS, device=DEVICE)
        timesteps = inverse_scheduler.timesteps
        # timesteps = self.model.scheduler.timesteps
        latents = latent
        # print(self.context.shape, "context shape")

        context_in = self.context

        if latent_2 is not None:
            uncond_e, cond_e = context_in.chunk(2)
            context_in = torch.cat([uncond_e, uncond_e, cond_e, cond_e], 0)
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)
        with self.model.progress_bar(total=NUM_DDIM_STEPS) as progress_bar:
            for i, t in enumerate(timesteps):

                if PROGRESS_BAR is not None:
                    PROGRESS_BAR(i / NUM_DDIM_STEPS, desc="Performing DDIM Inversion")

                

                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=context_in,
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                # noise_pred = noise_pred_cond

                # compute the previous noisy sample x_t -> x_t-1
                latents = inverse_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # print(latents.mean())
                all_latent.append(latents.detach())
                all_noise.append(noise_pred_cond.detach())
                progress_bar.update()

        return all_latent, all_noise
        # for i in range(NUM_DDIM_STEPS):
        #     t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
        #     noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
        #     latent = self.next_step(noise_pred, t, latent)
        #     all_latent.append(latent)
        # return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, image_2 = None):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        latent_2 = None
        if image_2 is not None:
            latent_2 = self.image2latent(image_2)

        ddim_latents, ddim_noise = self.ddim_loop(latent, latent_2)
        return image_rec, ddim_latents, ddim_noise

    def null_optimization(self, latents, num_inner_steps, epsilon, t_coords=None):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)

        with torch.enable_grad():
            for i in range(NUM_DDIM_STEPS):
                uncond_embeddings = uncond_embeddings.clone().detach()
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]
                t = self.model.scheduler.timesteps[i]
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                for j in range(num_inner_steps):

                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                    
                    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    
    #                 if 1:
                    # print("opt")
                    if t < -1:
                    # if t < int(0.5 * (1000)):
                        print(j)
                        loss = (nnf.mse_loss(latents_prev_rec, latent_prev) + self.null_text_optimization_transforms_loss(t, latent_cur, cond_embeddings, uncond_embeddings, latent_prev, noise_pred_cond, t_coords=t_coords)) / 2.0
                    else:
                        loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                    

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    bar.update()
                    if loss_item < epsilon + i * 2e-5:
                        break
                for j in range(j + 1, num_inner_steps):
                    bar.update()
                uncond_embeddings_list.append(uncond_embeddings[:1].detach())
                with torch.no_grad():
                    context = torch.cat([uncond_embeddings, cond_embeddings])
                    latent_cur = self.get_noise_pred(latent_cur, t, False, context)
            bar.close()
        return uncond_embeddings_list

    @torch.autocast("cuda", dtype=torch.half)
    def invert(self, image_gt, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False, t_coords = None, perform_inversion=True, image_2 = None):
        self.init_prompt(prompt)
        # ptp_utils.register_attention_control(self.model, None)
        # image_gt = load_512(image_path, *offsets)
        # plt.imshow(image_gt.astype(np.uint8))
        # plt.imshow(image_mask)
        # plt.show()
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents, ddim_noise = self.ddim_inversion(image_gt, image_2)

        if perform_inversion:
            if verbose:
                print("Null-text optimization...")
            uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon, t_coords = t_coords)
        else:
            uncond_embeddings = None
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings, ddim_latents, ddim_noise
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.controller = None


# Inference and our code

def reshape_transform_coords(transform_coords, in_mat = None, in_mat_shape = None):
    
    if in_mat is not None:
        s_z = in_mat.shape[-1]
    else:
        s_z = in_mat_shape[-1]
        

    # if s_z < transform_coords.shape[1]:
    #     t_coords = torch.nn.functional.interpolate(transform_coords.float().permute(0, -1, 1, 2), (s_z, s_z), mode="area").permute(0, 2, 3, 1).to(transform_coords.device).half()
    # else:

    smoothness_size = int(transform_coords.shape[1] / s_z)

    if smoothness_size % 2 == 0:
        smoothness_size = smoothness_size + 1

    if smoothness_size > 7:
        smoothness_size = 7
    elif smoothness_size < 3:
        smoothness_size = 3


    # print("Smoothing transform coords ", smoothness_size)
    # print(transform_coords.device)
    # t_coords_in = GaussianSmoothing(dim=2, channels = 3, kernel_size=smoothness_size)(transform_coords.permute(0, -1, 1, 2).to("cuda"))
    # t_coords_in = transform_coords

    # t_coords = T.Resize(size=(s_z, s_z), antialias=False, interpolation=T.InterpolationMode.BILINEAR)(t_coords_in).permute(0, 2, 3, 1).to(transform_coords.device)
    t_coords = T.Resize(size=(s_z, s_z), antialias=False, interpolation=T.InterpolationMode.BILINEAR)(transform_coords.permute(0, -1, 1, 2)).permute(0, 2, 3, 1).to(transform_coords.device)
    return t_coords

def reshape_attention_mask(mask, in_mat = None, in_mat_shape = None):
    
    s_m = int((mask.shape[-1]))
    
    if in_mat is not None:
        s_z = in_mat.shape[-1]
    else:
        s_z = in_mat_shape[-1]
        
    m = torch.reshape(mask, (mask.shape[0], mask.shape[1], s_m, s_m))

    # if m.shape[-1] > s_z:
    #     m = torch.nn.functional.interpolate(m.float(), (s_z, s_z), mode="area").half()
    # else:
    m = T.Resize(size=(s_z, s_z), antialias=False, interpolation=T.InterpolationMode.BILINEAR)(m)
    
    return m

def transform_mask_size(mask_in, t_mask_shape):
    
    mask = mask_in[0:1]#.detach()#.clone()
    mask = mask[:, None]
    mask = mask.tile((t_mask_shape[0], t_mask_shape[1], 1, 1)).type_as(t_mask)
    s_z = t_mask_shape[-1]
    mask = T.Resize(size=(s_z, s_z), antialias=False)(mask)
    #.tile((t_mask.shape))
    return mask


def perform_attention(q, k, v, scale, mask=None):

    

    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=True):
        
        out = F.scaled_dot_product_attention(q,k,v, scale=scale)

    return out

def binarize_tensor(t, thresh = 0.5):
    
    return (t > thresh) * 1.0

def get_base_edit_qkv(q, k, v, batch_size, coords_base = None, coords_edit = None, use_cfg = True):
    

    # print(q.shape, k.shape, v.shape)
    
    if use_cfg:
        h = q.shape[0] // (2 * batch_size)
        q = q.reshape(2 * batch_size, h, *q.shape[1:])
        k = k.reshape(2 * batch_size, h, *k.shape[1:])
        v = v.reshape(2 * batch_size, h, *v.shape[1:])

        # q_base, k_base, v_base = q[2:3], k[2:3], v[2:3] # 1, F, h*w, D
        # q_edit, k_edit, v_edit = q[3:], k[3:], v[3:] # b - 1, F, h*w, D
    else:
        h = q.shape[0] // (batch_size)
        q = q.reshape(batch_size, h, *q.shape[1:])
        k = k.reshape(batch_size, h, *k.shape[1:])
        v = v.reshape(batch_size, h, *v.shape[1:])

    # print(h, "h size", q.shape, k.shape, v.shape)

    
    if coords_base is not None:
        # print(coords_base)
        q_base, k_base, v_base = q[coords_base[0]:coords_base[1]], k[coords_base[0]:coords_base[1]], v[coords_base[0]:coords_base[1]] # 1, F, h*w, D
    
    if coords_edit is not None:
        # print(coords_edit)
        q_edit, k_edit, v_edit = q[coords_edit[0]:coords_edit[1]], k[coords_edit[0]:coords_edit[1]], v[coords_edit[0]:coords_edit[1]] # b - 1, F, h*w, D
    
    return q_base.detach(), k_base.detach(), v_base.detach(), q_edit, k_edit, v_edit


class CoordinateDistances:
    def __init__(self):
        self.coord_distance_dict = {}
        self.theta = torch.eye(3)[:2][None]
    
    def get_coord_distance(self, size, device="cuda"):
        if size not in self.coord_distance_dict:
            grid = torch.nn.functional.affine_grid(self.theta, (1, 1, size, size), align_corners=None) # 1, H, W, 2
            d = grid.reshape(1, -1, 2)
            dist = torch.sqrt(torch.sum(torch.square(d[:, :, None] - d[:, None]), -1) + 1e-12) # 1, hw, hw

            self.coord_distance_dict[size] = dist.to(device)

        return self.coord_distance_dict[size]

DISTANCE_CLASS = CoordinateDistances()



def smooth_attention_features(features):
    # features - b, h, n^2, D

    b, h_heads, n, D = features.shape
    im_size = int(np.sqrt(n))
    features_in = features.permute(0, 1, -1, 2).reshape(-1, 1, im_size, im_size) # B, 1, h, h
    features_out = GAUSSIAN_FEATURE_SMOOTHER(features_in)
    features_out = features_out.reshape(b, h_heads, D, n).permute(0, 1, -1, 2)

    return features_out


class AttentionGeometryEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t, transform_coords=None):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, transform_coords)
        return x_t
    
    def replace_cross_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = False, scale = None, mask = None, coords_base = None, coords_edit = None, old_attention_map = None, old_attention_out = None):
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        t_coords_m = reshape_transform_coords(transform_coords, in_mat_shape=self.image_mask.shape).tile(self.image_mask.shape[0], 1, 1, 1).type_as(q_base)
        
        if self.mask_new_warped is None:
            mask_new_warped = binarize_tensor(warp_grid_edit(self.image_mask[:, None], t_coords_m, padding_mode='zeros', align_corners=True, mode=MODE))
            self.mask_new_warped = mask_new_warped.detach()
        else:
            mask_new_warped = self.mask_new_warped.detach()


        amodal_mask = self.amodal_mask.detach()




        # print(amodal_mask.shape, mask_new_warped.shape, " amodal mask shape vs mask new warped shape")

        mask_warp = binarize_tensor(self.image_mask)[:, None]
        # print(amodal_mask.shape, mask_warp.shape)
        amodal_mask = amodal_mask.type_as(mask_warp)


        b, f = mask_new_warped.shape[:2]

        mask_new_warped = reshape_attention_mask(mask_new_warped, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]# b-1, 1, h, w
        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        amodal_mask = reshape_attention_mask(amodal_mask, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))
        amodal_mask = binarize_tensor(amodal_mask.type_as(mask_new_warped) - mask_new_warped).detach()
        
        mask_intersection = binarize_tensor((mask_new_warped + amodal_mask) * mask_warp, 0.5)
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp - mask_intersection), 0.5)


        h, w = mask_warp.shape[-2:]

        distance_grid = DISTANCE_CLASS.get_coord_distance(h).detach()
        # print(distance_grid.shape)
        # exit()
        
        # print(mask_warp.shape, h *w, 32**2)
        # if (h * w) >= 64**2:
        #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)
        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_new_warped) - binarize_tensor(mask_1_empty + mask_new_warped + amodal_mask))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
            self.mask_1_empty = mask_1_empty.detach()
        
        
        b, f, _, D = q_base.shape
        
        
        
        # Transform q_base to edit image        
        q_edit_base = q_base.permute(0, 1, -1, 2).reshape(b, f, D, h, w).reshape(-1, D, h, w)
        t_coords_q = reshape_transform_coords(transform_coords, in_mat_shape=q_edit_base.shape).tile(q_edit_base.shape[0], 1, 1, 1).type_as(q_edit_base)
        
        # Transform locations
        q_edit_base = q_edit_base * (1.0 - mask_new_warped) + mask_new_warped * warp_grid_edit(q_edit_base, t_coords_q, padding_mode='zeros', align_corners=True, mode=MODE)
        
        q_edit_base = q_edit_base.reshape(b, f, D, h, w).reshape(b, f, D, -1).permute(0, 1, -1, 2).reshape(-1, h*w, D)

        
        

        edit_base_att = compute_attention(q_edit_base, k_base.reshape(b*f, -1, D), scale, mask)
        edit_out = torch.bmm(edit_base_att, v_base.detach().reshape(b*f, -1, D))[None].detach()
        
        
        # edit_out = perform_attention(q_edit_base.reshape(b, f, -1, D), k_base.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale).reshape(b*f, -1 ,D).detach()[None]
        
        
        
        b, f, _, D = q_edit.shape

        # if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):
        #     replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask, fg_mask_warp = mask_new_warped, fg_mask = mask_warp)
        #     replace_out = torch.bmm(replace_out_att, v_base.detach().reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D

        # else:
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        replace_out = torch.bmm(replace_out_att, v_base.detach().reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        # replace_out = perform_attention(q_edit.reshape(b, f, -1, D), k_edit.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale)
        


        # if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):


        #     replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        #     replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_inpaint = mask_1_empty[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]
        mask_edit = mask_new_warped[0, 0].reshape(-1)[None, None, :, None]
        amodal_mask = amodal_mask[0, 0].reshape(-1)[None, None, :, None]


        # print(mask_inpaint.shape, replace_out.shape)
        b, f, _, D = replace_out.shape



        if self.use_cfg and self.store_attention_maps:
            # print(q_edit.shape)
            self.attn_store(replace_out_att, is_cross=True, place_in_unet = place_in_unet)





        if mask_inpaint.shape[2] >= 32 ** 2 and (not self.use_cfg):
            interpolated_features, interpolation_weights = interpolate_from_mask(edit_out, mask_edit, distance_grid)
            


            # print(interpolated_features.shape, edit_out.shape)
            interpolated_features[:, :, mask_edit[0, 0, :, 0] > 0.5] = edit_out[:, :, mask_edit[0, 0, :, 0] > 0.5].detach().type_as(interpolated_features)
            # print(interpolated_features.shape, edit_out.shape)
            interpolated_features = smooth_attention_features(interpolated_features)
            # exit()
            


            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint * torch.ones_like(replace_out.detach())) + 1e-8))
            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint.expand_as(replace_out)) + 1e-8))

            # dissimilar_loss = get_correlation_loss_stitch(compute_attention(q_edit.reshape(b*f, -1, D), q_base.detach().reshape(b*f, -1, D), scale, mask), mask_inpaint, mask_wo_edit, mask_inpaint)



            # print(sim_loss)

            # with torch.no_grad():
            #     # Do not get min distance from regions that are in the background
            #     distance_weights = distance_grid.type_as(mask_wo_edit) + 100 * mask_wo_edit[:1, :1, :, 0]
            #     distance_weights = 1.0 - torch.exp(-2 * torch.min(distance_weights, -1).values) # 1, hw
            # sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit * distance_weights[:, None, :, None]) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach()) * distance_weights[:, None, :, None]) + 1e-8))

            # print(sim_loss)
            # exit()
            sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach())) + 1e-8))
            # movement_loss = (torch.sum((1 -  torch.exp((-torch.abs(edit_out.detach() - replace_out)))) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out.detach())) + 1e-8))


            movement_loss = torch.sum(torch.abs(edit_out.detach() - replace_out) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out.detach())) + 1e-8)
            
            
            with torch.no_grad():
                if self.use_cfg:
                    ah = q.shape[0] // (2 * self.batch_size)
                else:
                    ah = q.shape[0] // (self.batch_size)

                base_att = old_attention_map[coords_base[0] * ah: coords_base[1] * ah]
                # base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)

            correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], base_att.permute(0, -1, 1).detach())

            correlation_inpaint = correlation * mask_inpaint[..., 0]
            correlation_wo_edit = correlation * mask_wo_edit[..., 0]

            m_c_inpaint = torch.max(correlation_inpaint, -1)
            m_c_wo_edit = torch.max(correlation_wo_edit, -1)

            p_correlation_inpaint, d_inpaint = m_c_inpaint.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_inpaint.indices]
            p_correlation_wo_edit, d_wo_edit = m_c_wo_edit.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_wo_edit.indices]


            with torch.no_grad():
                d_weight = torch.exp(-d_wo_edit.detach())

            dissimilar_loss = torch.sum(d_weight.detach() * (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))) / (torch.sum(mask_inpaint) * f + 1e-8)
            del m_c_inpaint, m_c_wo_edit


            # correlation_amodal = torch.bmm(replace_out_att[:, amodal_mask[0, 0, :, 0] > 0.5], base_att[:, mask_inpaint[0, 0, :, 0] > 0.5].permute(0, -1, 1).detach())

            # correlation_amodal_loss = torch.sum(-torch.log(torch.max(correlation_amodal, -1).values + 1e-8)) / (torch.sum(torch.ones_like(correlation_amodal)) + 1e-8)



            amodal_loss = (torch.sum((torch.abs(interpolated_features.detach() - replace_out)) * interpolation_weights[..., None] * amodal_mask) / (torch.sum(interpolation_weights[..., None] * amodal_mask * torch.ones_like(replace_out.detach())) + 1e-8))     

            # amodal_loss = 0.0 * movement_loss
            # correlation_amodal = torch.bmm(replace_out_att[:, amodal_mask[0, 0, :, 0] > 0.5], edit_base_att[:, :, ].permute(0, -1, 1).detach())


            # amodal_loss = 


            if mask_inpaint.shape[2] <= 32 ** 2:
                amodal_loss = 0.0 * movement_loss



            dissociate_loss, att_loss, dissociate_loss_2 = 0, 0, 0

            smoothness_loss, _, _ = get_smoothness_loss(replace_out)

            # if mask_inpaint.shape[2] >= 64 ** 2:
            #     smoothness_loss, _, _ = get_smoothness_loss(replace_out)
            # else: 
            #     smoothness_loss = 0.0 * movement_loss

            lw = self.loss_weight_dict["cross"]
            self.loss += lw["sim"] * sim_loss + lw["movement"] * movement_loss + lw["removal"] * dissimilar_loss + lw["smoothness"] * smoothness_loss + lw["amodal"] * amodal_loss

            loss_log_dict = {"sim": sim_loss, "movement": movement_loss, "removal": dissimilar_loss, "smoothness": smoothness_loss, "amodal": amodal_loss}
            self.loss_log_dict["cross"] = update_loss_log_dict(self.loss_log_dict["cross"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1
            
            

            # self.loss += (0.0 * dissociate_loss + 0.0 * att_loss + 10.0 * sim_loss + 10.0 * movement_loss + 0.0 *  dissociate_loss_2 + 8.0 * dissimilar_loss) / 3



        
        # edit_out[:, :, amodal_mask[0, 0, :, 0] > 0.5] = interpolated_features[:, :, amodal_mask[0, 0, :, 0] > 0.5].detach().type_as(edit_out)
        # mask_edit = binarize_tensor(mask_edit + amodal_mask)
        
        if (self.cur_step < int(self.num_steps * self.obj_edit_step)):
            out_return = edit_out.detach() * (mask_edit) + replace_out * (1.0 - mask_edit) #mask_inpaint + replace_out * mask_wo_edit
        # elif mask_inpaint.shape[2] <= 16 ** 2:
        #     out_return = edit_out.detach() * (mask_edit) + replace_out * (1.0 - mask_edit)
        else:
            out_return = replace_out
            # print("identity")
            # out_return = replace_out_identity * (1.0 - mask_edit) + edit_out.detach() * mask_edit
            # out_return = replace_out_identity * (mask_edit) + replace_out_identity * mask_inpaint + replace_out * mask_wo_edit
        
        return out_return

        
        
        
    def replace_self_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = None, scale = None, mask = None, coords_base = None, coords_edit = None, old_attention_map = None, old_attention_out = None):
        
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        t_coords_m = reshape_transform_coords(transform_coords, in_mat_shape=self.image_mask.shape).tile(self.image_mask.shape[0], 1, 1, 1).type_as(q_base)
        

        if self.mask_new_warped is None:
            mask_new_warped = binarize_tensor(warp_grid_edit(self.image_mask[:, None], t_coords_m, padding_mode='zeros', align_corners=True, mode=MODE))
            self.mask_new_warped = mask_new_warped.detach()

        else:
            mask_new_warped = self.mask_new_warped.detach()


        amodal_mask = self.amodal_mask.detach()

        # amodal_mask = binarize_tensor(self.amodal_mask.type_as(mask_new_warped) - mask_new_warped).detach()

        # print(amodal_mask.shape, mask_new_warped.shape, " amodal mask shape vs mask new warped shape")
            

        mask_warp = binarize_tensor(self.image_mask.type_as(q_base))[:, None]
        # print(amodal_mask.shape, mask_warp.shape)
        # exit()

        amodal_mask = amodal_mask.type_as(mask_warp)
        b, f = mask_new_warped.shape[:2]

        mask_new_warped = reshape_attention_mask(mask_new_warped, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]# b-1, 1, h, w
        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        amodal_mask = reshape_attention_mask(amodal_mask, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))
        amodal_mask = binarize_tensor(amodal_mask.type_as(mask_new_warped) - mask_new_warped).detach()
        
        
        
        mask_intersection = binarize_tensor((mask_new_warped + amodal_mask) * mask_warp, 0.5)
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp - mask_intersection), 0.5)
        
        h, w = mask_warp.shape[-2:]
        distance_grid = DISTANCE_CLASS.get_coord_distance(h).detach()
        
        # if (h * w) >= 64**2:
        #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)
        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_new_warped) - (mask_1_empty + mask_new_warped))
        
        
        
        b, f, _, D = q_base.shape
        # Transform q_base to edit image        
        q_edit_base = q_base.permute(0, 1, -1, 2).reshape(b, f, D, h, w).reshape(-1, D, h, w)
        t_coords_q = reshape_transform_coords(transform_coords, in_mat_shape=q_edit_base.shape).tile(q_edit_base.shape[0], 1, 1, 1).type_as(q_edit_base)
        
        # Transform locations
        q_edit_base = q_edit_base * (1.0 - mask_new_warped) + mask_new_warped * warp_grid_edit(q_edit_base, t_coords_q, padding_mode='zeros', align_corners=True, mode=MODE)
        
        q_edit_base = q_edit_base.reshape(b, f, D, h, w).reshape(b, f, D, -1).permute(0, 1, -1, 2).reshape(-1, h*w, D)
        
        
        edit_base_att = compute_attention(q_edit_base, k_base.reshape(b*f, -1, D), scale, mask)
        edit_out = torch.bmm(edit_base_att, v_base.detach().reshape(b*f, -1, D))[None].detach()
        
        
        b, f, _, D = q_edit.shape

        # q_edit_adain = adain(q_edit.reshape(b*f, -1, D), q_base.detach().reshape(b*f, -1, D))

        # if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask, fg_mask_warp = mask_new_warped, fg_mask = mask_warp, inpaint_mask = mask_wo_edit)
        replace_out = torch.bmm(replace_out_att, v_base.detach().reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        
        # else:

        #     replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask)
        #     replace_out = torch.bmm(replace_out_att, v_base.detach().reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        

        if self.use_cfg and self.store_attention_maps: 
            # print(q_edit.shape)

            self.attn_store(replace_out_att, is_cross=False, place_in_unet = place_in_unet)

        # edit_out = perform_attention(q_edit_base.reshape(b, f, -1, D), k_base.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale).reshape(b*f, -1 ,D).detach()[None]
        
        
        
        # b, f, _, D = q_edit.shape
        # replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        # replace_out = torch.bmm(replace_out_att, v_base.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        # replace_out = perform_attention(q_edit.reshape(b, f, -1, D), k_edit.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale)


        # if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):

        #     replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        #     replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_inpaint = mask_1_empty[0, 0].reshape(-1)[None, None, :, None]
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]
        mask_edit = mask_new_warped[0, 0].reshape(-1)[None, None, :, None]
        amodal_mask = amodal_mask[0, 0].reshape(-1)[None, None, :, None]

        
        if mask_inpaint.shape[2] >= 32 ** 2:
            self.mask_inpaint = mask_1_empty[0, 0].detach().clone()
        
        

        # if (self.cur_step < int(self.num_steps * 0.6)):
        #     edit_out = smooth_attention_features(edit_out.detach())



        if mask_inpaint.shape[2] >= 32 ** 2 and (not self.use_cfg):
            interpolated_features, interpolation_weights = interpolate_from_mask(edit_out, mask_edit, distance_grid)

            interpolated_features[:, :, mask_edit[0, 0, :, 0] > 0.5] = edit_out[:, :, mask_edit[0, 0, :, 0] > 0.5].detach().type_as(interpolated_features)
            # print(interpolated_features.shape, edit_out.shape)
            interpolated_features = smooth_attention_features(interpolated_features)

            # sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach())) + 1e-8))

            
            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint * torch.ones_like(replace_out.detach())) + 1e-8))

            with torch.no_grad():

                if self.use_cfg:
                    ah = q.shape[0] // (2 * self.batch_size)
                else:
                    ah = q.shape[0] // (self.batch_size)

                base_att = old_attention_map[coords_base[0] * ah: coords_base[1] * ah]
                # base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)
                # print(base_att.shape)


            # print(mask_inpaint.shape, replace_out_att.shape)
            # print(torch.sum((mask_inpaint[0, 0, :, 0] > 0.5) * 1.0))

            # exit()
            correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], base_att.permute(0, -1, 1).detach())

            correlation_inpaint = correlation * mask_inpaint[..., 0]
            correlation_wo_edit = correlation * mask_wo_edit[..., 0]

            # p_correlation_inpaint = torch.max(correlation_inpaint, -1).values
            # p_correlation_wo_edit = torch.max(correlation_wo_edit, -1).values

            # dissimilar_loss = torch.sum(-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4)) / (torch.sum(mask_inpaint) * f + 1e-8)

            m_c_inpaint = torch.max(correlation_inpaint, -1)
            m_c_wo_edit = torch.max(correlation_wo_edit, -1)

            p_correlation_inpaint, d_inpaint = m_c_inpaint.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_inpaint.indices]
            p_correlation_wo_edit, d_wo_edit = m_c_wo_edit.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_wo_edit.indices]

            # print(d_inpaint.max(), d_inpaint.min(), d_wo_edit.max(), d_wo_edit.min())
            
            # print("before: ", torch.log(p_correlation_inpaint + 1e-4).min(), torch.log(p_correlation_inpaint + 1e-4).max())
            # print("after: ", torch.exp(-15 * d_wo_edit.detach()).max(), (torch.exp(-15 * d_wo_edit.detach()) * torch.log(p_correlation_inpaint + 1e-4)).min(), (torch.exp(-15 * d_wo_edit.detach()) * torch.log(p_correlation_inpaint + 1e-4)).max())

            # d_before = (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))
            # d_after = (-torch.log(p_correlation_wo_edit + 1e-4) + torch.exp(-15 * d_wo_edit.detach()) * torch.log(p_correlation_inpaint + 1e-4))
            
            
            # print("+++++++++++++++++++++++++++++++++++++++++++++++")
            # print("before: ", d_before.min(), d_before.max())
            # print("after: ", d_after.min(), d_after.max())

            

            with torch.no_grad():
                d_weight = torch.exp(-d_wo_edit.detach())
                
            dissimilar_loss = torch.sum(d_weight.detach() * (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))) / (torch.sum(mask_inpaint) * f + 1e-8)

            del m_c_inpaint, m_c_wo_edit


            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint.expand_as(replace_out)) + 1e-8))
            # dissimilar_loss = get_correlation_loss_stitch(replace_out_att, mask_inpaint, mask_wo_edit, mask_inpaint)

            # with torch.no_grad():
            #     # Do not get min distance from regions that are in the background
            #     distance_weights = distance_grid.type_as(mask_wo_edit) + 100 * mask_wo_edit[:1, :1, :, 0]
            #     distance_weights = 1.0 - torch.exp(-2 * torch.min(distance_weights, -1).values) # 1, hw
                
            # sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit * distance_weights[:, None, :, None]) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach()) * distance_weights[:, None, :, None]) + 1e-8))

            sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit.expand_as(replace_out)) + 1e-8))

            # movement_loss = (torch.sum((1 -  torch.exp((-torch.abs(edit_out.detach() - replace_out)))) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out.detach())) + 1e-8))
            movement_loss = torch.sum(torch.abs(edit_out.detach() - replace_out) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out.detach())) + 1e-8)


            

            # correlation_amodal = torch.bmm(replace_out_att[:, amodal_mask[0, 0, :, 0] > 0.5], base_att[:, mask_inpaint[0, 0, :, 0] > 0.5].permute(0, -1, 1).detach())

            # correlation_amodal_loss = torch.sum(-torch.log(torch.max(correlation_amodal, -1).values + 1e-8)) / (torch.sum(torch.ones_like(correlation_amodal)) + 1e-8)


            # interpolated_features, interpolation_weights = interpolate_from_mask(edit_out, mask_edit, distance_grid)
            amodal_loss = (torch.sum((torch.abs(interpolated_features.detach() - replace_out)) * interpolation_weights[..., None] * amodal_mask) / (torch.sum(interpolation_weights[..., None] * amodal_mask * torch.ones_like(replace_out.detach())) + 1e-8))        


            # print(interpolated_features.shape, interpolation_weights.shape, replace_out.shape)
            # print(amodal_loss)


            if mask_inpaint.shape[2] <= 32 ** 2:
                amodal_loss = 0.0 * movement_loss

            
            
            dissociate_loss, att_loss, dissociate_loss_2 = 0, 0, 0
            smoothness_loss, _, _ = get_smoothness_loss(replace_out)

            # if mask_inpaint.shape[2] >= 64 ** 2:
            #     smoothness_loss, _, _ = get_smoothness_loss(replace_out)
            # else:
            #     smoothness_loss = 0.0 * movement_loss

            lw = self.loss_weight_dict["self"]
            self.loss += lw["sim"] * sim_loss + lw["movement"] * movement_loss + lw["removal"] * dissimilar_loss + lw["smoothness"] * smoothness_loss + lw["amodal"] * amodal_loss

            
            loss_log_dict = {"sim": sim_loss, "movement": movement_loss, "removal": dissimilar_loss, "smoothness": smoothness_loss, "amodal": amodal_loss}
            self.loss_log_dict["self"] = update_loss_log_dict(self.loss_log_dict["self"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1
            
            

            # self.loss += (0.0 * dissociate_loss + 0.0 * att_loss + 55.0 * sim_loss + 20.0 * movement_loss + 0.0 * dissociate_loss_2 + 13.0 * dissimilar_loss) / 3

        # mask_edit = binarize_tensor(mask_edit + amodal_mask)

        # edit_out[:, :, amodal_mask[0, 0, :, 0] > 0.5] = interpolated_features[:, :, amodal_mask[0, 0, :, 0] > 0.5].detach().type_as(edit_out)
        # mask_edit = binarize_tensor(mask_edit + amodal_mask)

        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = edit_out.detach() * (mask_edit) + replace_out * (1.0 - mask_edit) #mask_inpaint + replace_out * mask_wo_edit
        # elif mask_inpaint.shape[2] <= 16 ** 2:
        #     out_return = edit_out.detach() * (mask_edit) + replace_out * (1.0 - mask_edit)
        else:
            out_return = replace_out

            # print("identity")
            # out_return = replace_out_identity * (1.0 - mask_edit) + edit_out.detach() * mask_edit
            # out_return = replace_out_identity * (mask_edit) + replace_out_identity * mask_inpaint + replace_out * mask_wo_edit

        return out_return

    def initialize_loss_log_dict(self):

        self.loss_log_dict = {"self": {"sim": 0.0, "movement": 0.0, "removal": 0.0, "smoothness": 0.0},
                             "cross": {"sim": 0.0, "movement": 0.0, "removal": 0.0, "smoothness": 0.0}, 
                             "num_layers": 0}

    
    def forward(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale = None, mask = None):


        if self.use_cfg:
            h = q.shape[0] // (2 * self.batch_size)
        else:
            h = q.shape[0] // self.batch_size


        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            attn = compute_attention(q[:self.coords_base[-1] * h], k[:self.coords_base[-1] * h], scale, mask)
            out = torch.bmm(attn, v[:self.coords_base[-1] * h])
        else:
            attn = compute_attention(q, k, scale, mask)
            out = torch.bmm(attn, v)

        


        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                out_edit = self.replace_cross_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty = False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit, old_attention_map = attn, old_attention_out = out)
                
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

            else:
                out_edit = self.replace_self_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty=False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit, old_attention_map = attn, old_attention_out = out)
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])
        

        return out
    
    
    def initialize_default_loss_weights(self):
        self.loss_weight_dict = self.default_loss_weights

        # old
        # self.loss_weight_dict = {"self": {"sim": 40, "movement": 1.5, "removal": 1.67, "smoothness": 45.0},
        #                         "cross": {"sim": 25, "movement": 1.34, "removal": 1.6, "smoothness": 20.0}}
        # Increase movement loss
        # reduce background preservation loss
        # From remover : )
        # self.loss_weight_dict = {"self": {"sim": 0.97, "scale": 75, "smoothness": 85.0},
        # "cross": {"sim": 0.92, "scale": 45.0, "smoothness": 45.0}}


    # def store_default_loss_weights(self):
    #     self.default

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer = None,
                local_blend = None, controller = None, image_mask = None, empty_scale = 0.2, use_all = True, obj_edit_step = 0.0):
        super(AttentionGeometryEdit, self).__init__()
        if equalizer is not None:
            eq = get_equalizer(prompts[1], equalizer["words"], equalizer["values"])
            self.equalizer = eq.to(DEVICE)
        self.prev_controller = controller
        self.last_cross_mask = None
        
        if image_mask is not None:
            image_mask = torch.from_numpy(image_mask[None])
            # print(image_mask.shape)
            image_mask = image_mask.tile((len(prompts), 1, 1))
            self.image_mask = image_mask
        self.thre = 0.00001
        self.empty_scale = empty_scale
        self.use_all = use_all
        self.loss = 0.0

        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(DEVICE)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.mask_inpaint = None
        self.obj_edit_step = obj_edit_step
        self.num_steps = num_steps

        self.mask_new_warped = None
        self.mask_wo_edit = None
        self.mask_1_empty = None
        self.amodal_mask = None

        self.coords_base = (2, 3)
        self.coords_edit = (3, 4)
        self.use_cfg = True


        # self.loss_weight_dict = {"self": {"sim": 0.74, "scale": 25.0, "removal": 4.34, "smoothness": 0.0},
        #                         "cross": {"sim": 0.5, "scale": 6.67, "removal": 2.67, "smoothness": 0.0}}
        

        self.default_loss_weights = {"self": {"sim": 110, "movement": 13.5, "removal": 1.67, "smoothness": 35.0, "amodal": 80.5},
                                "cross": {"sim": 60, "movement": 6.34, "removal": 1.6, "smoothness": 20.0, "amodal": 3.5}}
        self.loss_log_dict = None
        self.loss_weight_dict = None
        self.initialize_loss_log_dict()
        self.store_attention_maps = False
        self.initialize_default_loss_weights()



class AttentionGeometryRemover(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t, transform_coords=None):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, transform_coords)
        return x_t
    
    def replace_cross_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = False, scale = None, mask = None, coords_base = None, coords_edit = None, old_attention_map = None, old_attention_out = None):
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)

        if self.use_cfg:
            h_head = q.shape[0] // (2 * self.batch_size)
        else:
            h_head = q.shape[0] // self.batch_size
        
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        
        mask_warp = binarize_tensor(self.image_mask)[:, None]
        b, f = mask_warp.shape[:2]

        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp), 0.5)

        h, w = mask_warp.shape[-2:]
        distance_grid = DISTANCE_CLASS.get_coord_distance(h).detach()

        # if (h * w) >= 64**2:
        # #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)

        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_warp) - (mask_1_empty))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
            self.mask_1_empty = mask_1_empty.detach()
        
        
        b, f, _, D = q_base.shape

        edit_base_att = old_attention_map[coords_base[0] * h_head:coords_base[1]*h_head].detach()
        edit_out = old_attention_out[coords_base[0] * h_head:coords_base[1]*h_head][None].detach()
        # edit_base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)
        # edit_out = torch.bmm(edit_base_att, v_base.reshape(b*f, -1, D))[None].detach()
        
        
        b, f, _, D = q_edit.shape
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask)
        replace_out = torch.bmm(replace_out_att, v_base.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        



        # print(edit_base_att.shape, replace_out_att.shape)

        
        # if self.cur_step < int(self.num_steps * self.obj_edit_step):
        if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):

            replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
            replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_inpaint = mask_1_empty[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]


        b, f, _, D = replace_out.shape



        if mask_inpaint.shape[2] >= 32 ** 2 and (not self.use_cfg):



            # dissimilar_loss = - (torch.sum(torch.min((torch.sum(torch.abs(replace_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5][:, :, :, None] - edit_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5][:, :, None].detach()), -1)), -1).values) / (torch.sum(mask_inpaint.expand_as(replace_out)) * 8 + 1e-8)) # b, 8, M_#

            # print(dissimilar_loss, " cross")
            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint.expand_as(replace_out)) + 1e-8))
            sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit.expand_as(replace_out)) + 1e-8))

            # correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], edit_base_att.permute(0, -1, 1).detach())
            
            # # loss_correlation = torch.sum(correlation  * mask_inpaint[..., 0] - correlation  * mask_wo_edit[..., 0]) / (torch.sum(mask_inpaint) * h * w * f + 1e-8)
            # loss_correlation = torch.sum(torch.max(correlation * mask_inpaint[..., 0], -1).values - torch.max(correlation  * mask_wo_edit[..., 0], -1).values) / (torch.sum(mask_inpaint) * f + 1e-8)

            correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], edit_base_att.permute(0, -1, 1).detach())

            correlation_inpaint = correlation * mask_inpaint[..., 0]
            correlation_wo_edit = correlation * mask_wo_edit[..., 0]

            # p_correlation_inpaint = torch.max(correlation_inpaint, -1).values
            # p_correlation_wo_edit = torch.max(correlation_wo_edit, -1).values

            # loss_correlation = torch.sum(-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4)) / (torch.sum(mask_inpaint) * f + 1e-8)

            m_c_inpaint = torch.max(correlation_inpaint, -1)
            m_c_wo_edit = torch.max(correlation_wo_edit, -1)

            p_correlation_inpaint, d_inpaint = m_c_inpaint.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_inpaint.indices]
            p_correlation_wo_edit, d_wo_edit = m_c_wo_edit.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_wo_edit.indices]


            with torch.no_grad():
                d_weight = torch.exp(-d_wo_edit.detach())

            loss_correlation = torch.sum(d_weight.detach() * (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))) / (torch.sum(mask_inpaint) * f + 1e-8)
            del m_c_inpaint, m_c_wo_edit

            smoothness_loss, _, _ = get_smoothness_loss(replace_out)
            # print(smoothness_loss)
            
            

    # b, h*w, h*w
            # print(loss_correlation, "Cross")        
            dissociate_loss, att_loss, dissociate_loss_2, movement_loss = 0, 0, 0, 0.0

            lw = self.loss_weight_dict["cross"]
            self.loss += (lw["sim"] * sim_loss + lw["removal"]* loss_correlation) + lw["smoothness"] * smoothness_loss


            loss_log_dict = {"sim": sim_loss, "removal": loss_correlation, "smoothness": smoothness_loss}
            self.loss_log_dict["cross"] = update_loss_log_dict(self.loss_log_dict["cross"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1

            # self.loss += 0.5 * (0.0 * dissociate_loss + 0.0 * att_loss + 200.0 * sim_loss + 0.0 * movement_loss + 0.0 *  dissociate_loss_2 + 0.0 * dissimilar_loss + 20.0 * loss_correlation + 200 * smoothness_loss) / 3


        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = replace_out * mask_inpaint + replace_out * mask_wo_edit
        else:
            out_return = replace_out_identity * mask_inpaint + replace_out * mask_wo_edit
        

        return out_return

        
        
        
    def replace_self_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = None, scale = None, mask = None, coords_base = None, coords_edit = None, old_attention_map = None, old_attention_out = None):
        
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        if self.use_cfg:
            h_head = q.shape[0] // (2 * self.batch_size)
        else:
            h_head = q.shape[0] // self.batch_size
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        mask_warp = binarize_tensor(self.image_mask)[:, None]
        b, f = mask_warp.shape[:2]
        
        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp), 0.5)
        h, w = mask_warp.shape[-2:]
        distance_grid = DISTANCE_CLASS.get_coord_distance(h).detach()

        # if (h * w) >= 64**2:
        # #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)

        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_warp) - (mask_1_empty))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
            self.mask_1_empty = mask_1_empty.detach()
        
        
        b, f, _, D = q_base.shape


        edit_base_att = old_attention_map[coords_base[0] * h_head:coords_base[1]*h_head].detach()
        edit_out = old_attention_out[coords_base[0] * h_head:coords_base[1]*h_head][None].detach()
        # edit_base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)
        # edit_out = torch.bmm(edit_base_att, v_base.reshape(b*f, -1, D))[None].detach()
        
        
        b, f, _, D = q_edit.shape
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask)
        replace_out = torch.bmm(replace_out_att, v_base.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        

        # if self.cur_step < int(self.num_steps * self.obj_edit_step):


        if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):
            replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
            replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_inpaint = mask_1_empty[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]


        b, f, _, D = replace_out.shape


        # print(edit_out.shape, replace_out.shape, mask_inpaint.shape)
        # print(replace_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5].shape)

        # dissimilar_loss = - (torch.sum(torch.min((torch.sum(torch.abs(replace_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5][:, :, :, None] - edit_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5][:, :, None].detach()), -1)), -1).values) / (torch.sum(mask_inpaint.expand_as(replace_out)) * 8 + 1e-8)) # b, 8, M_#


        if mask_inpaint.shape[2] >= 32 ** 2 and (not self.use_cfg):
            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint.expand_as(replace_out)) + 1e-8))
            sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit.expand_as(replace_out)) + 1e-8))
            
            # print(dissimilar_loss)

            correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], edit_base_att.permute(0, -1, 1).detach())

            correlation_inpaint = correlation * mask_inpaint[..., 0]
            correlation_wo_edit = correlation * mask_wo_edit[..., 0]

            # p_correlation_inpaint = torch.max(correlation_inpaint, -1).values
            # p_correlation_wo_edit = torch.max(correlation_wo_edit, -1).values

            # loss_correlation = torch.sum(-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4)) / (torch.sum(mask_inpaint) * f + 1e-8)


            m_c_inpaint = torch.max(correlation_inpaint, -1)
            m_c_wo_edit = torch.max(correlation_wo_edit, -1)

            p_correlation_inpaint, d_inpaint = m_c_inpaint.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_inpaint.indices]
            p_correlation_wo_edit, d_wo_edit = m_c_wo_edit.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_wo_edit.indices]


            with torch.no_grad():
                d_weight = torch.exp(-d_wo_edit.detach())

            loss_correlation = torch.sum(d_weight.detach() * (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))) / (torch.sum(mask_inpaint) * f + 1e-8)
            del m_c_inpaint, m_c_wo_edit
            
            # loss_correlation = torch.sum(torch.max(correlation * mask_inpaint[..., 0] / (torch.sum(correlation * mask_inpaint[..., 0], -1, keepdims=True) + 1e-8), -1).values - torch.max(correlation  * mask_wo_edit[..., 0] / (torch.sum(correlation * mask_wo_edit[..., 0], -1, keepdims=True) + 1e-8), -1).values) / (torch.sum(mask_inpaint) * f + 1e-8)

            # print(loss_correlation, "Self")        

            # print(replace_out.shape, " replace out")

            smoothness_loss, _, _ = get_smoothness_loss(replace_out)
            # print(smoothness_loss)

            # exit()

            dissociate_loss, att_loss, dissociate_loss_2, movement_loss = 0, 0, 0, 0.0


            lw = self.loss_weight_dict["self"]
            self.loss += (lw["sim"] * sim_loss + lw["removal"]* loss_correlation) + lw["smoothness"] * smoothness_loss

            loss_log_dict = {"sim": sim_loss, "removal": loss_correlation, "smoothness": smoothness_loss}
            self.loss_log_dict["self"] = update_loss_log_dict(self.loss_log_dict["self"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1


            # self.loss += 0.5 * (0.0 * dissociate_loss + 0.0 * att_loss + 365.0 * sim_loss + 0.0 * movement_loss + 0.0 *  dissociate_loss_2 + 0.0 * dissimilar_loss + 35.0 * loss_correlation + 330 * smoothness_loss) / 3


        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = replace_out * mask_inpaint + replace_out * mask_wo_edit
        else:
            out_return = replace_out_identity * mask_inpaint + replace_out * mask_wo_edit
        

        return out_return    


    def forward(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale = None, mask = None):


        if self.use_cfg:
            h = q.shape[0] // (2 * self.batch_size)
        else:
            h = q.shape[0] // self.batch_size


        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            attn = compute_attention(q[:self.coords_base[-1] * h], k[:self.coords_base[-1] * h], scale, mask)
            out = torch.bmm(attn, v[:self.coords_base[-1] * h])
        else:
            attn = compute_attention(q, k, scale, mask)
            out = torch.bmm(attn, v)


        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                out_edit = self.replace_cross_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty = False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit, old_attention_map = attn, old_attention_out = out)
                
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

            else:
                out_edit = self.replace_self_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty=False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit, old_attention_map = attn, old_attention_out = out)
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

        return out
    

    def initialize_loss_log_dict(self):

        self.loss_log_dict = {"self": {"sim": 0.0, "removal": 0.0, "smoothness": 0.0},
                             "cross": {"sim": 0.0, "removal": 0.0, "smoothness": 0.0}, 
                             "num_layers": 0}
    
    def initialize_default_loss_weights(self):
        self.loss_weight_dict = self.default_loss_weights


    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer = None,
                local_blend = None, controller = None, image_mask = None, empty_scale = 0.2, use_all = True, obj_edit_step = 0.0):
        super(AttentionGeometryRemover, self).__init__()
        if equalizer is not None:
            eq = get_equalizer(prompts[1], equalizer["words"], equalizer["values"])
            self.equalizer = eq.to(DEVICE)
        self.prev_controller = controller
        self.last_cross_mask = None
        
        if image_mask is not None:
            image_mask = torch.from_numpy(image_mask[None])
            image_mask = image_mask.tile((len(prompts), 1, 1))
            self.image_mask = torch_dilate(image_mask[:, None], 5)[:, 0]
        self.thre = 0.00001
        self.empty_scale = empty_scale
        self.use_all = use_all
        self.loss = 0.0

        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(DEVICE)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.mask_inpaint = None
        self.obj_edit_step = obj_edit_step
        self.num_steps = num_steps

        self.mask_new_warped = None
        self.mask_wo_edit = None
        self.mask_1_empty = None
        self.amodal_mask = None

        self.coords_base = (2, 3)
        self.coords_edit = (3, 4)
        self.use_cfg = True

        self.loss_weight_dict = None
        # self.default_loss_weights = {"self": {"sim": 73, "removal": 2.25, "smoothness": 85.0},
        # "cross": {"sim": 41.4, "removal": 3.6, "smoothness": 45.0}}

        self.default_loss_weights = {"self": {"sim": 110.0, "removal": 3.6, "smoothness": 35.0},
        "cross": {"sim": 60.0, "removal": 3.6, "smoothness": 20.0}}

        # self.default_loss_weights = {"self": {"sim": 110, "movement": 13.5, "removal": 1.67, "smoothness": 35.0},
        #                         "cross": {"sim": 60, "movement": 6.34, "removal": 1.6, "smoothness": 20.0}}
        self.initialize_default_loss_weights()
        self.loss_log_dict = None
        self.initialize_loss_log_dict()
        

class AttentionGeometryStitchSingle(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t, transform_coords=None):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, transform_coords)
        return x_t
    
    def replace_cross_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = False, scale = None, mask = None, coords_base = None, coords_edit = None):
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        
        mask_warp = binarize_tensor(self.image_mask)[:, None]
        b, f = mask_warp.shape[:2]

        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp), 0.5)

        h, w = mask_warp.shape[-2:]
        # if (h * w) >= 64**2:
        # #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)

        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_warp) - (mask_1_empty))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
            self.mask_1_empty = mask_1_empty.detach()
        
        
        b, f, _, D = q_base.shape

        edit_base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)
        edit_out = torch.bmm(edit_base_att, v_base.reshape(b*f, -1, D))[None].detach()
        
        
        b, f, _, D = q_edit.shape
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask)
        replace_out = torch.bmm(replace_out_att, v_base.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        



        # print(edit_base_att.shape, replace_out_att.shape)

        
        # if self.cur_step < int(self.num_steps * self.obj_edit_step):
        if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):

            replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
            replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_stitch = mask_1_empty[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]


        b, f, _, D = replace_out.shape



        if mask_stitch.shape[2] >= 32 ** 2:

            dist = torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)
            sim_loss_out = (torch.sum(dist[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach())) + 1e-8))
            movement_loss_out = (torch.sum(dist[..., None] * mask_stitch) / (torch.sum(mask_stitch * torch.ones_like(replace_out.detach())) + 1e-8))

            del dist

            smoothness_loss, _, _ = get_smoothness_loss(replace_out)
      
            dissociate_loss, att_loss, dissociate_loss_2, movement_loss = 0, 0, 0, 0.0

            lw = self.loss_weight_dict["cross"]
            self.loss += lw["scale_out"] * (lw["sim_out"] * sim_loss_out + (1.0 - lw["sim_out"]) * movement_loss_out) + lw["smoothness"] * smoothness_loss


            loss_log_dict = {"sim_out": sim_loss_out, "movement_out": movement_loss_out, "smoothness": smoothness_loss}
            self.loss_log_dict["cross"] = update_loss_log_dict(self.loss_log_dict["cross"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1

            # self.loss += 0.5 * (0.0 * dissociate_loss + 0.0 * att_loss + 200.0 * sim_loss + 0.0 * movement_loss + 0.0 *  dissociate_loss_2 + 0.0 * dissimilar_loss + 20.0 * loss_correlation + 200 * smoothness_loss) / 3


        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = replace_out * mask_stitch + replace_out * mask_wo_edit
        else:
            out_return = replace_out_identity * mask_stitch + replace_out_identity * mask_wo_edit
        

        return out_return

        
        
        
    def replace_self_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = None, scale = None, mask = None, coords_base = None, coords_edit = None):
        
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        
        mask_warp = binarize_tensor(self.image_mask)[:, None]
        b, f = mask_warp.shape[:2]

        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp), 0.5)

        h, w = mask_warp.shape[-2:]
        # if (h * w) >= 64**2:
        # #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)

        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_warp) - (mask_1_empty))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
            self.mask_1_empty = mask_1_empty.detach()
        
        
        b, f, _, D = q_base.shape

        edit_base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)
        edit_out = torch.bmm(edit_base_att, v_base.reshape(b*f, -1, D))[None].detach()
        
        
        b, f, _, D = q_edit.shape
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask)
        replace_out = torch.bmm(replace_out_att, v_base.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        



        # print(edit_base_att.shape, replace_out_att.shape)

        
        # if self.cur_step < int(self.num_steps * self.obj_edit_step):
        if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):

            replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
            replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_stitch = mask_1_empty[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]


        b, f, _, D = replace_out.shape



        if mask_stitch.shape[2] >= 32 ** 2:

            dist = torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)
            sim_loss_out = (torch.sum(dist[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach())) + 1e-8))
            movement_loss_out = (torch.sum(dist[..., None] * mask_stitch) / (torch.sum(mask_stitch * torch.ones_like(replace_out.detach())) + 1e-8))

            del dist

            smoothness_loss, _, _ = get_smoothness_loss(replace_out)
      
            dissociate_loss, att_loss, dissociate_loss_2, movement_loss = 0, 0, 0, 0.0

            lw = self.loss_weight_dict["self"]
            self.loss += lw["scale_out"] * (lw["sim_out"] * sim_loss_out + (1.0 - lw["sim_out"]) * movement_loss_out) + lw["smoothness"] * smoothness_loss


            loss_log_dict = {"sim_out": sim_loss_out, "movement_out": movement_loss_out, "smoothness": smoothness_loss}
            self.loss_log_dict["self"] = update_loss_log_dict(self.loss_log_dict["self"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1

            # self.loss += 0.5 * (0.0 * dissociate_loss + 0.0 * att_loss + 200.0 * sim_loss + 0.0 * movement_loss + 0.0 *  dissociate_loss_2 + 0.0 * dissimilar_loss + 20.0 * loss_correlation + 200 * smoothness_loss) / 3


        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = replace_out * mask_stitch + replace_out * mask_wo_edit
        else:
            out_return = replace_out_identity * mask_stitch + replace_out_identity * mask_wo_edit
        

        return out_return

    def forward(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale = None, mask = None):

        attn = compute_attention(q, k, scale, mask)

        if self.use_cfg:
            h = attn.shape[0] // (2 * self.batch_size)
        else:
            h = attn.shape[0] // self.batch_size

        out = torch.bmm(attn, v)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                out_edit = self.replace_cross_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty = False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit)
                
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

            else:
                out_edit = self.replace_self_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty=False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit)
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

        return out
    

    def initialize_loss_log_dict(self):

        self.loss_log_dict = {"self": {"sim_out": 0.0, "movement_out": 0.0, "smoothness": 0.0},
                             "cross": {"sim_out": 0.0, "movement_out": 0.0, "smoothness": 0.0}, 
                             "num_layers": 0}
    
    def initialize_default_loss_weights(self):
        self.loss_weight_dict = self.default_loss_weights


    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer = None,
                local_blend = None, controller = None, image_mask = None, empty_scale = 0.2, use_all = True, obj_edit_step = 0.0):
        super(AttentionGeometryStitchSingle, self).__init__()
        if equalizer is not None:
            eq = get_equalizer(prompts[1], equalizer["words"], equalizer["values"])
            self.equalizer = eq.to(DEVICE)
        self.prev_controller = controller
        self.last_cross_mask = None
        
        if image_mask is not None:
            image_mask = torch.from_numpy(image_mask[None])
            # print(image_mask.shape)
            image_mask = image_mask.tile((len(prompts), 1, 1))
            self.image_mask = image_mask
        self.thre = 0.00001
        self.empty_scale = empty_scale
        self.use_all = use_all
        self.loss = 0.0

        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(DEVICE)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.mask_inpaint = None
        self.obj_edit_step = obj_edit_step
        self.num_steps = num_steps

        self.mask_new_warped = None
        self.mask_wo_edit = None
        self.mask_1_empty = None
        self.amodal_mask = None

        self.coords_base = (2, 3)
        self.coords_edit = (3, 4)
        self.use_cfg = True


        self.default_loss_weights = {"self": {"sim": 0.97, "scale": 75, "smoothness": 85.0},
        "cross": {"sim": 0.92, "scale": 45.0, "smoothness": 45.0}}

        self.initialize_default_loss_weights()
        self.loss_log_dict = None
        self.initialize_loss_log_dict()


def calc_mean_std(feat, eps: float = 1e-5, dim = -2):
    feat_std = (feat.var(dim=dim, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=dim, keepdims=True)
    return feat_mean, feat_std


def adain(feat, feat_2, dim=-2):
    feat_mean, feat_std = calc_mean_std(feat, dim=dim)
    feat_2_mean, feat_2_std = calc_mean_std(feat_2, dim=dim)

    # print("feat 1: ", feat_mean, feat_std)
    # print("feat 2: ", feat_2_mean,.shape feat_2_std.shape)
    # feat_style_mean = expand_first(feat_mean)
    # feat_style_std = expand_first(feat_std)
    feat_1 = (feat - feat_mean) / feat_std
    feat_1 = feat_1 * feat_2_std + feat_2_mean
    return feat_1


class AttentionGeometryStitch(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t, transform_coords=None):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, transform_coords)
        return x_t
    
    def replace_cross_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = False, scale = None, mask = None, coords_base = None, coords_edit = None):
        
        
          # First latent is for background
        # Second latent is for the stitch image
        # third latent is for the edit image
        # 2, f, h*w, d
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)


        
        q_base_bg, k_base_bg, v_base_bg = q_base[:1], k_base[:1], v_base[:1]
        q_base_stitch, k_base_stitch, v_base_stitch = q_base[1:2], k_base[1:2], v_base[1:2]

        # print(torch.abs(q_base_bg - q_base_stitch).sum(), torch.abs(k_base_bg - k_base_stitch).sum(), torch.abs(v_base_bg - v_base_stitch).sum(), " difference 1")

        self.image_mask = self.image_mask.type_as(q_base).detach()
        t_coords_m = reshape_transform_coords(transform_coords, in_mat_shape=self.image_mask.shape).tile(self.image_mask.shape[0], 1, 1, 1).type_as(q_base)
        
        if self.mask_new_warped is None:
            mask_new_warped = binarize_tensor(warp_grid_edit(self.image_mask[:, None], t_coords_m, padding_mode='zeros', align_corners=True, mode=MODE))
            self.mask_new_warped = mask_new_warped.detach()
        else:
            mask_new_warped = self.mask_new_warped.detach()
        mask_warp = binarize_tensor(self.image_mask)[:, None]


        b, f = mask_new_warped.shape[:2]

        mask_new_warped = (reshape_attention_mask(mask_new_warped, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:])# b-1, 1, h, w
        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        h, w = mask_warp.shape[-2:]
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_new_warped) - (mask_new_warped))


        # print("saving")
        # plt.imsave("./debug.png", mask_wo_edit[0, 0].detach().cpu().numpy(), cmap="gray")
        # plt.imsave("./debug_2.png", mask_new_warped[0, 0].detach().cpu().numpy(), cmap="gray")
        # print(mask_wo_edit.shape, mask_wo_edit.min(), mask_wo_edit.max(), mask_new_warped.min(), mask_new_warped.max(), mask_new_warped.shape)

        # exit()

        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
        
        
        b, f, _, D = q_base_stitch.shape
        
        
        
        # Transform q_base to edit image        
        q_edit_base_stitch = q_base_stitch.permute(0, 1, -1, 2).reshape(b, f, D, h, w).reshape(-1, D, h, w)
        t_coords_q = reshape_transform_coords(transform_coords, in_mat_shape=q_edit_base_stitch.shape).tile(q_edit_base_stitch.shape[0], 1, 1, 1).type_as(q_edit_base_stitch)
        

        # print(q_edit_base_stitch.shape, t_coords_q.shape, mask_new_warped.shape)
        # exit()
        # Transform locations
        q_edit_base_stitch = q_edit_base_stitch * (1.0 - mask_new_warped[:1]) + mask_new_warped[:1] * warp_grid_edit(q_edit_base_stitch, t_coords_q, padding_mode='zeros', align_corners=True, mode=MODE)
        
        q_edit_base_stitch = q_edit_base_stitch.reshape(b, f, D, h, w).reshape(b, f, D, -1).permute(0, 1, -1, 2).reshape(-1, h*w, D)

        # print(torch.abs(q_edit_base_stitch - q_base_stitch.reshape(b*f, -1, D)).sum(), " difference 2")

        
        
        
        # Stitching transforms
        edit_base_stitch_att = compute_attention(q_edit_base_stitch, k_base_stitch.reshape(b*f, -1, D), scale, mask)
        edit_out_stitch = torch.bmm(edit_base_stitch_att, v_base_stitch.reshape(b*f, -1, D))[None].detach()
        
        # Background transforms
        base_bg_att = compute_attention(q_base_bg.reshape(b*f, -1, D), k_base_bg.reshape(b*f, -1, D), scale, mask)
        base_bg = torch.bmm(base_bg_att, v_base_bg.reshape(b*f, -1, D)).reshape(b*f, -1, D)[None].detach()
        

        
        
        # Edit image
        b, f, _, D = q_edit.shape
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        replace_out_att_bg = compute_attention(q_edit.reshape(b*f, -1, D), k_base_bg.detach().reshape(b*f, -1, D), scale, mask)
        # Edited background
        replace_out_bg = torch.bmm(replace_out_att_bg, v_base_bg.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        # replace_out = perform_attention(q_edit.reshape(b, f, -1, D), k_edit.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale)
        
        # Edited Stitch

        # V_r = bg, V_t = fg    
        with torch.no_grad():
            k_base_stitch_adain = adain(k_base_stitch.detach().reshape(b*f, -1, D), k_base_bg.detach().reshape(b*f, -1, D))
            k_base_bg_stitch = torch.cat([k_base_bg.detach().reshape(b*f, -1, D), k_base_stitch_adain], -2)
            v_base_bg_stitch = torch.cat([v_base_bg.detach().reshape(b*f, -1, D), v_base_stitch.detach().reshape(b*f, -1, D)], -2)

        replace_out_att_stitch = compute_attention(adain(q_edit.reshape(b*f, -1, D), q_base_stitch.detach().reshape(b*f, -1, D)), k_base_bg_stitch.detach(), scale, mask)
        replace_out_stitch = torch.bmm(replace_out_att_stitch, v_base_bg_stitch.detach()).reshape(b, f, -1, D) # 1, f, h*w, D

        # print("works")

        if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):

            replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
            replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out_stitch = edit_out_stitch.tile(replace_out_stitch.shape[0], 1, 1, 1) # b, f, h*w, D
        base_bg = base_bg.tile(replace_out_stitch.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_edit = mask_new_warped[0, 0].reshape(-1)[None, None, :, None]

        mask_old_wo_edit = (torch.ones_like(mask_warp) - mask_warp)[0, 0].reshape(-1)[None, None, :, None]
        mask_warp = mask_warp[0, 0].reshape(-1)[None, None, :, None]

        b, f, _, D = replace_out_stitch.shape



        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            # out_return = base_bg
            out_return = replace_out_bg * mask_wo_edit + edit_out_stitch * mask_edit

        else:
            out_return = replace_out_identity * (mask_edit) + replace_out_bg * mask_wo_edit
        
        if mask_wo_edit.shape[2] >= 32 ** 2:
            

            h_im = int(np.sqrt(mask_wo_edit.shape[2]))

            with torch.no_grad():
                distance_grid = DISTANCE_CLASS.get_coord_distance(h_im).detach()
                d_weights_grid = 1.0 - torch.exp(-(torch.min(distance_grid + 1000 * mask_wo_edit[:1, :1, :, 0], -1).values))

            b, f, _, _ = replace_out_bg.shape
            # Background similarity loss
            sim_loss_out = torch.sum(torch.sum(base_bg_att.detach() * (torch.log(base_bg_att.detach() + 1e-8) - torch.log(replace_out_att_bg + 1e-8)), -1) * mask_wo_edit[0, 0, :, 0][None] * d_weights_grid) / (torch.sum(b * f * mask_wo_edit[0, 0, :, 0][None]) + 1e-8)

            # movement_loss_out = torch.sum(torch.sum(edit_base_stitch_att.detach() * (torch.log(edit_base_stitch_att.detach() + 1e-8) - torch.log(replace_out_att_stitch + 1e-8)), -1) * mask_edit[0, 0, :, 0][None]) / (torch.sum(b * f * mask_edit[0, 0, :, 0][None]) + 1e-8)
            # sim_loss_out = (torch.sum((torch.sum((torch.abs(base_bg.detach() - replace_out_bg)), -1))[..., None] * mask_wo_edit * d_weights_grid[None, :, :, None]) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out_bg.detach())) + 1e-8))
            movement_loss_out = torch.sum(torch.abs(edit_out_stitch.detach() - replace_out_stitch) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out_stitch.detach())) + 1e-8)
            # sim_loss = torch.sum(torch.sum(base_bg_att.detach() * (torch.log(base_bg_att.detach() + 1e-8) - torch.log(replace_out_att_bg + 1e-8)), -1) * mask_wo_edit[0, 0, :, 0][None]) / (torch.sum(b * f * mask_wo_edit[0, 0, :, 0][None]) + 1e-8)

            # sim_loss = torch.sum(-torch.log(torch.sum(base_bg_att * replace_out_att_bg, -1) + 1e-8) * mask_wo_edit[0, 0, :, 0][None]) / (edit_base_stitch_att.shape[0] * torch.sum(mask_wo_edit[0, 0, :, 0]) + 1e-8)


            # Foreground movement loss
            # movement_loss = torch.sum(torch.sum(edit_base_stitch_att.detach() * (torch.log(edit_base_stitch_att.detach() + 1e-8) - torch.log(replace_out_att_stitch + 1e-8)), -1) * mask_edit[0, 0, :, 0][None]) / (torch.sum(b * f * mask_edit[0, 0, :, 0][None]) + 1e-8)
            # movement_loss = torch.sum(-torch.log(torch.sum(edit_base_stitch_att * replace_out_att_stitch, -1) + 1e-8) * mask_edit[0, 0, :, 0][None]) / (edit_base_stitch_att.shape[0] * torch.sum(mask_edit[0, 0, :, 0]) + 1e-8)
            # movement_loss = get_correlation_loss_stitch(compute_attention(q_edit.reshape(b*f, -1, D), q_base_stitch.reshape(b*f, -1, D), scale, mask), mask_old_wo_edit, mask_warp, mask_edit)
            
            smoothness_loss, _, _ = get_smoothness_loss(out_return)
            
            movement_loss = movement_loss_out * 0.0
            sim_loss = sim_loss_out * 0.0
            lw = self.loss_weight_dict["cross"]
            self.loss += (lw["sim"] * sim_loss + lw["movement"] * movement_loss) + lw["smoothness"] * smoothness_loss
            self.loss += (lw["sim_out"] * sim_loss_out + lw["movement_out"] * movement_loss_out)

            loss_log_dict = {"sim": sim_loss, "movement": movement_loss, "smoothness": smoothness_loss, "sim_out": sim_loss_out, "movement_out": movement_loss_out}
            self.loss_log_dict["cross"] = update_loss_log_dict(self.loss_log_dict["cross"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1

            # self.loss += (5.0 * sim_loss + 5.0 * movement_loss) / 3
            # self.loss += (10.0 * sim_loss + 10.0 * movement_loss) / 3

        
        return out_return

        
        
        
    def replace_self_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = None, scale = None, mask = None, coords_base = None, coords_edit = None):
        
        
          # First latent is for background
        # Second latent is for the stitch image
        # third latent is for the edit image
        # 2, f, h*w, d
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        q_base_bg, k_base_bg, v_base_bg = q_base[:1], k_base[:1], v_base[:1]
        q_base_stitch, k_base_stitch, v_base_stitch = q_base[1:2], k_base[1:2], v_base[1:2]


        # print(q_base_bg.shape, k_base_bg.shape, k_base_stitch.shape, v_base_bg.shape, v_base_stitch.shape)

        self.image_mask = self.image_mask.type_as(q_base).detach()
        t_coords_m = reshape_transform_coords(transform_coords, in_mat_shape=self.image_mask.shape).tile(self.image_mask.shape[0], 1, 1, 1).type_as(q_base)
        
        if self.mask_new_warped is None:
            mask_new_warped = binarize_tensor(warp_grid_edit(self.image_mask[:, None], t_coords_m, padding_mode='zeros', align_corners=True, mode=MODE))
            self.mask_new_warped = mask_new_warped.detach()
        else:
            mask_new_warped = self.mask_new_warped.detach()
        mask_warp = binarize_tensor(self.image_mask)[:, None]


        b, f = mask_new_warped.shape[:2]

        mask_new_warped = (reshape_attention_mask(mask_new_warped, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:])# b-1, 1, h, w
        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        h, w = mask_warp.shape[-2:]
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_new_warped) - (mask_new_warped))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
        
        
        b, f, _, D = q_base_stitch.shape
        
        
        
        # Transform q_base to edit image        
        q_edit_base_stitch = q_base_stitch.permute(0, 1, -1, 2).reshape(b, f, D, h, w).reshape(-1, D, h, w)
        t_coords_q = reshape_transform_coords(transform_coords, in_mat_shape=q_edit_base_stitch.shape).tile(q_edit_base_stitch.shape[0], 1, 1, 1).type_as(q_edit_base_stitch)
        

        # print(q_edit_base_stitch.shape, t_coords_q.shape, mask_new_warped.shape)
        # exit()
        # Transform locations
        q_edit_base_stitch = q_edit_base_stitch * (1.0 - mask_new_warped[:1]) + mask_new_warped[:1] * warp_grid_edit(q_edit_base_stitch, t_coords_q, padding_mode='zeros', align_corners=True, mode=MODE)
        
        q_edit_base_stitch = q_edit_base_stitch.reshape(b, f, D, h, w).reshape(b, f, D, -1).permute(0, 1, -1, 2).reshape(-1, h*w, D)

        
        
        # Stitching transforms
        edit_base_stitch_att = compute_attention(q_edit_base_stitch, k_base_stitch.reshape(b*f, -1, D), scale, mask)
        edit_out_stitch = torch.bmm(edit_base_stitch_att, v_base_stitch.reshape(b*f, -1, D))[None].detach()
        
        # Background transforms
        base_bg_att = compute_attention(q_base_bg.reshape(b*f, -1, D), k_base_bg.reshape(b*f, -1, D), scale, mask)
        base_bg = torch.bmm(base_bg_att, v_base_bg.reshape(b*f, -1, D)).reshape(b*f, -1, D)[None].detach()
        

        
        
        # Edit image
        b, f, _, D = q_edit.shape
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)

        replace_out_att_bg = compute_attention(q_edit.reshape(b*f, -1, D), k_base_bg.detach().reshape(b*f, -1, D), scale, mask)
        # Edited background
        replace_out_bg = torch.bmm(replace_out_att_bg, v_base_bg.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        # replace_out = perform_attention(q_edit.reshape(b, f, -1, D), k_edit.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale)
        
        # Edited Stitch
        # replace_out_att_stitch = compute_attention(q_edit.reshape(b*f, -1, D), k_base_stitch.detach().reshape(b*f, -1, D), scale, mask)

        # replace_out_att_stitch = compute_attention(q_edit_base_stitch, k_edit.reshape(b*f, -1, D), scale, mask)

        # replace_out_stitch = torch.bmm(replace_out_att_stitch, v_base_stitch.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D


        with torch.no_grad():
            k_base_stitch_adain = adain(k_base_stitch.detach().reshape(b*f, -1, D), k_base_bg.detach().reshape(b*f, -1, D))
            k_base_bg_stitch = torch.cat([k_base_bg.detach().reshape(b*f, -1, D), k_base_stitch_adain], -2)
            v_base_bg_stitch = torch.cat([v_base_bg.detach().reshape(b*f, -1, D), v_base_stitch.detach().reshape(b*f, -1, D)], -2)

        replace_out_att_stitch = compute_attention(adain(q_edit.reshape(b*f, -1, D), q_base_stitch.detach().reshape(b*f, -1, D)), k_base_bg_stitch.detach(), scale, mask)
        replace_out_stitch = torch.bmm(replace_out_att_stitch, v_base_bg_stitch.detach()).reshape(b, f, -1, D) # 1, f, h*w, D


        if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):

            replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
            replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out_stitch = edit_out_stitch.tile(replace_out_stitch.shape[0], 1, 1, 1) # b, f, h*w, D
        base_bg = base_bg.tile(replace_out_stitch.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_edit = mask_new_warped[0, 0].reshape(-1)[None, None, :, None]


        b, f, _, D = replace_out_stitch.shape




        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = replace_out_bg * mask_wo_edit + edit_out_stitch * mask_edit
        else:
            out_return = replace_out_identity * (mask_edit) + replace_out_bg * mask_wo_edit



        if mask_wo_edit.shape[2] >= 32 ** 2:

            mask_old_wo_edit = (torch.ones_like(mask_warp) - mask_warp)[0, 0].reshape(-1)[None, None, :, None]
            mask_warp = mask_warp[0, 0].reshape(-1)[None, None, :, None]

            b, f, _, _ = replace_out_bg.shape

            h_im = int(np.sqrt(mask_wo_edit.shape[2]))

            with torch.no_grad():
                distance_grid = DISTANCE_CLASS.get_coord_distance(h_im).detach()
                d = torch.min(distance_grid + 1000.0 * mask_wo_edit[:1, :1, :, 0], -1).values
                d_weights_grid = 1.0 - torch.exp(-d)

            b, f, _, _ = replace_out_bg.shape
            # Background similarity loss
            sim_loss_out = torch.sum(torch.sum(base_bg_att.detach() * (torch.log(base_bg_att.detach() + 1e-8) - torch.log(replace_out_att_bg + 1e-8)), -1) * mask_wo_edit[0, 0, :, 0][None] * d_weights_grid) / (torch.sum(b * f * mask_wo_edit[0, 0, :, 0][None]) + 1e-8)
            
            # sim_loss_out = (torch.sum((torch.sum((torch.abs(base_bg.detach() - replace_out_bg)), -1))[..., None] * mask_wo_edit * d_weights_grid[None, :, :, None]) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out_bg.detach())) + 1e-8))

            # Background similarity loss
            # sim_loss_out = (torch.sum(1-torch.exp(-torch.sum((torch.abs(base_bg.detach() - replace_out_bg)), -1))[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit.expand_as(replace_out_bg)) + 1e-8))

            # sim_loss = torch.sum(-torch.log(torch.sum(base_bg_att * replace_out_att_bg, -1) + 1e-8) * mask_wo_edit[0, 0, :, 0][None]) / (edit_base_stitch_att.shape[0] * torch.sum(mask_wo_edit[0, 0, :, 0]) + 1e-8)

            # Foreground movement loss
            # movement_loss_out = torch.sum(torch.abs(edit_out_stitch.detach() - replace_out_stitch) * mask_edit) / (torch.sum(mask_edit.expand_as(replace_out_stitch)) + 1e-8)
            # movement_loss = torch.sum(torch.sum(edit_base_stitch_att.detach() * (torch.log(edit_base_stitch_att.detach() + 1e-8) - torch.log(replace_out_att_stitch + 1e-8)), -1) * mask_edit[0, 0, :, 0][None]) / (torch.sum(b * f * mask_edit[0, 0, :, 0][None]) + 1e-8)
            # movement_loss = get_correlation_loss_stitch(q_edit.reshape(b*f, -1, D), q_base_stitch.detach(), mask_old_wo_edit, mask_warp)
            # movement_loss = get_correlation_loss_stitch(replace_out_att_stitch, mask_old_wo_edit, mask_warp, mask_new_warped)
            # movement_loss = get_correlation_loss_stitch(compute_attention(q_edit.reshape(b*f, -1, D), q_base_stitch.reshape(b*f, -1, D), scale, mask), mask_old_wo_edit, mask_warp, mask_edit)
            
            # movement_loss = torch.sum(-torch.log(torch.sum(edit_base_stitch_att * replace_out_att_stitch, -1) + 1e-8) * mask_edit[0, 0, :, 0][None]) / (edit_base_stitch_att.shape[0] * torch.sum(mask_edit[0, 0, :, 0]) + 1e-8)

            # sim_loss_out = (torch.sum((torch.sum((torch.abs(base_bg.detach() - replace_out_bg)), -1))[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out_bg.detach())) + 1e-8))


            # movement_loss_out = torch.sum(torch.sum(edit_base_stitch_att.detach() * (torch.log(edit_base_stitch_att.detach() + 1e-8) - torch.log(replace_out_att_stitch + 1e-8)), -1) * mask_edit[0, 0, :, 0][None]) / (torch.sum(b * f * mask_edit[0, 0, :, 0][None]) + 1e-8)

            movement_loss_out = torch.sum(torch.abs(edit_out_stitch.detach() - replace_out_stitch) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out_stitch.detach())) + 1e-8)

            smoothness_loss, _, _ = get_smoothness_loss(out_return)
            movement_loss = movement_loss_out * 0.0
            sim_loss = sim_loss_out * 0.0
            lw = self.loss_weight_dict["self"]
            # self.loss += (55.0 * sim_loss + 20.0 * movement_loss) / 3

            self.loss += (lw["sim"] * sim_loss + lw["movement"] * movement_loss) + lw["smoothness"] * smoothness_loss
            self.loss += (lw["sim_out"] * sim_loss_out + lw["movement_out"] * movement_loss_out)

            loss_log_dict = {"sim": sim_loss, "movement": movement_loss, "smoothness": smoothness_loss, "sim_out": sim_loss_out, "movement_out": movement_loss_out}
            self.loss_log_dict["self"] = update_loss_log_dict(self.loss_log_dict["self"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1
            # self.loss += 4 * (15.0 * sim_loss + 13.0 * movement_loss) / 3

        return out_return
    
    def forward(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale = None, mask = None):
        
        attn = compute_attention(q, k, scale, mask)

        if self.use_cfg:
            h = attn.shape[0] // (2 * self.batch_size)
        else:
            h = attn.shape[0] // self.batch_size

        out = torch.bmm(attn, v)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                out_edit = self.replace_cross_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty = False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit)
                
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

            else:
                out_edit = self.replace_self_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty=False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit)
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

        return out
    
    def initialize_loss_log_dict(self):

        self.loss_log_dict = {"self": {"sim": 0.0, "movement": 0.0, "smoothness": 0.0, "sim_out": 0.0, "movement_out": 0.0},
                             "cross": {"sim": 0.0, "movement": 0.0, "smoothness": 0.0, "sim_out": 0.0, "movement_out": 0.0}, 
                             "num_layers": 0}

    def initialize_default_loss_weights(self):
        self.loss_weight_dict = self.default_loss_weights

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer = None,
                local_blend = None, controller = None, image_mask = None, empty_scale = 0.2, use_all = True, obj_edit_step = 0.0):
        super(AttentionGeometryStitch, self).__init__()

        self.last_cross_mask = None
        
        if image_mask is not None:
            image_mask = torch.from_numpy(image_mask[None])
            # print(image_mask.shape)
            image_mask = image_mask.tile((len(prompts), 1, 1))
            self.image_mask = image_mask
        self.thre = 0.00001
        self.empty_scale = empty_scale
        self.use_all = use_all
        self.loss = 0.0

        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(DEVICE)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.mask_inpaint = None
        self.obj_edit_step = obj_edit_step
        self.num_steps = num_steps

        self.mask_new_warped = None
        self.mask_wo_edit = None
        self.mask_1_empty = None
        self.amodal_mask = None

        self.coords_base = (3, 5)
        self.coords_edit = (5, 6)
        self.use_cfg = True


        self.default_loss_weights = {"self": {"sim": 28, "movement": 12.0, "smoothness": 0.0, "sim_out": 0.0, "movement_out": 0.0},
                                "cross": {"sim": 2.5, "movement": 2.5, "smoothness": 0.0, "sim_out": 0.0, "movement_out": 0.0}}
        
        self.initialize_default_loss_weights()

        self.loss_log_dict = None
        self.initialize_loss_log_dict()

        


@torch.autocast("cuda", dtype=torch.half)
def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, transform_coords=None, use_cfg=True, return_noise = False):
 
    if use_cfg:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred_out = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    else:
        latents = latents#.to(memory_format=torch.channels_last)
        model.unet = model.unet#.to(memory_format=torch.channels_last)
        noise_pred_out = model.unet(latents, t, encoder_hidden_states=context)["sample"]

    latents_out = model.scheduler.step(noise_pred_out, t, latents, eta=0.0)["prev_sample"]
    latents_out = controller.step_callback(latents_out, transform_coords)
    SPLATTER.clear_cache()

    if return_noise:
        return latents_out, noise_pred_out

    return latents_out
@torch.autocast("cuda", dtype=torch.half)
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

# @torch.no_grad()
def image2latent(image, model, mask = None):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            
            if mask is not None:
                
                if type(mask) is np.ndarray:
                    mask_in = torch.from_numpy(mask).float()
                    image = image * (mask_in[..., None] < 0.5)
                    image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                else:
                    mask_in = mask
                    image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE).tile(mask.shape[0], 1, 1, 1)
                    image = image * (mask_in < 0.5).to(DEVICE)
                    
            else:
                image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                 
            
            latents = model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents




def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, 4, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  4, height // 8, width // 8).to(model.device)
    return latent, latents

def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float, mask = None, context = None, scaler = None, optimizer = None) -> torch.Tensor:
    """ Update the latent according to the computed loss. """
    
    
    if context is not None:

        # loss_new = scaler.scale(loss)
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients
        loss_new  = loss
        if scaler is not None:
            loss_new = scaler.scale(loss)    
        



        if optimizer is None:
        # torch.autograd.backward(loss_new, inputs=[latents, context], retain_graph=False)
            grads = torch.autograd.grad(loss_new, [latents, context], retain_graph = False)
        
        else:
            print("[INFO]: Using PyTorch Optimizer")
            optimizer.zero_grad()
            loss_new.backward()
            if scaler is not None:
                print("[INFO]: Using Scaler")
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            

            latents_in, context_new = optimizer.param_groups[0]["params"]
            latents_out = latents_in.detach().clone()
            context_out = context_new.detach().clone()

            latents_out[torch.logical_not(torch.isfinite(latents_out))] = latents[torch.logical_not(torch.isfinite(latents_out))].detach()
            context_out[torch.logical_not(torch.isfinite(context_out))] = context[torch.logical_not(torch.isfinite(context_out))].detach() 
            # latents_in = ((1.0 * torch.isfinite(latents_in)) * latents_in + (1.0 - (1.0 * torch.isfinite(latents_in))) * latents.detach()).detach()
            # context_new = ((1.0 * torch.isfinite(context_new)) * context_new + (1.0 - (1.0 * torch.isfinite(context_new))) * context.detach()).detach()

            return latents_out, context_out

        # if scaler is not None:
        #     inv_scale = 1./scaler.get_scale()
        #     grads = [p * inv_scale for p in grads]
        # context_grad = context.grad
        # grad_cond = latents.grad

        grad_cond = grads[0]
        context_grad = grads[1]

        context_grad = torch.nan_to_num(context_grad, posinf=0.0, neginf=0.0, nan=0.0)
        grad_cond = torch.nan_to_num(grad_cond, posinf=0.0, neginf=0.0, nan=0.0)#latents.grad
        # if scaler is not None:
        #     scaler.update()

    context_new = None


    if grad_cond is not None:
        
        # if mask is not None and edit_type != "geometry_stitch":
        if mask is not None:
            mask_inpaint = reshape_attention_mask(mask[None, None].type_as(latents), in_mat_shape = latents[-1:].shape)       

            latents = torch.cat([latents[:-1], latents[-1:].detach() - mask_inpaint[-1:] * step_size * grad_cond[-1:]], 0)
            latents = torch.cat([latents[:-1], latents[-1:].detach() - (1.0 - mask_inpaint[-1:]) * step_size * grad_cond[-1:]], 0)

        # else:
            # print("latent grads")
        # print("Running latent grad", grad_cond[:-1].min(), grad_cond[:-1].max())
        # print("Running latent grad", grad_cond[-1:].min(), grad_cond[-1:].max())
        else:
            latents = torch.cat([latents[:-1], latents[-1:].detach() - step_size * grad_cond[-1:]], 0)

        if context is not None:

            # print("Running context grad", context_grad[:-1].min(), context_grad[:-1].max())
            # print("Running context grad", context_grad[-1:].min(), context_grad[-1:].max())

            context_new = torch.cat([context[:-1], context[-1:].detach() - step_size * context_grad[-1:]], 0)

            # context_new = torch.cat([context[:1], context[1:2].detach().clone() - step_size * context_grad[1:2], context[2:]], 0)
                
        
        
    else:
        print("no grad found")
    return latents, context_new





class EditProcessor:
    def __init__(self, transform_coords, controller, place_in_unet = "down", perform_edit = True, coords_base=(2, 3), coords_edit=(3,4), use_cfg=True):
        # self.attention_probs = None
        self.transform_coords = transform_coords
        self.place_in_unet = place_in_unet
        self.perform_edit = perform_edit
        self.controller = controller

        self.controller.use_cfg = use_cfg
        self.controller.coords_base = coords_base
        self.controller.coords_edit = coords_edit

    
    # @torch.enable_grad()
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask = None,
        temb = None,
        scale: float = 1.0,
    ) -> torch.Tensor:

        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        # print(attn.scale, attn.rescale_output_factor)
        # args = (scale,)
        # args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)
        is_cross = True
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross = False
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)


        
        if self.perform_edit:
            hidden_states = self.controller(query, key, value, is_cross = is_cross, place_in_unet = self.place_in_unet, transform_coords = self.transform_coords, scale = attn.scale,  mask = None)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def adain_latents(latent, latent_ref):

    b, f, h, w = latent.shape
    latent_in = latent.reshape(latent.shape[0], f, -1)
    latent_ref_in = latent_ref.reshape(latent_ref.shape[0], f, -1)

    latent_adain = adain(latent_in, latent_ref_in, dim=-1).reshape(b, f, h, w)

    
    return latent_adain
    
def register_attention_control_diffusers(model, controller, transform_coords = None):
    
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        # print(name)

        cross_att_count += 1
        attn_processor = EditProcessor(transform_coords, controller, place_in_unet)
        # if name.startswith("up_blocks.3") or name.startswith("up_blocks.2"):
        # if name.startswith("up_blocks.3") or name.startswith("up_blocks.2") or name.startswith("down_blocks.0") or name.startswith("down_blocks.1"):
        # if 
        if 1:
            attn_procs[name] = attn_processor
        else:
            attn_procs[name] = VanillaAttentionProcessor()

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


def set_attn_processor_for_edit(model, perform_edit=True, coords_base=(2,3), coords_edit = (3,4), use_cfg = True):
    # Setting attn processor to enable / disable editing
    for name in model.unet.attn_processors.keys():

        # if name.startswith("up_blocks.3") or name.startswith("up_blocks.2"):
        if 1:
        # if name.startswith("up_blocks.3") or name.startswith("up_blocks.2") or name.startswith("down_blocks.0") or name.startswith("down_blocks.1"):

            model.unet.attn_processors[name].perform_edit = perform_edit
            model.unet.attn_processors[name].controller.coords_base = coords_base
            model.unet.attn_processors[name].controller.coords_edit = coords_edit
            model.unet.attn_processors[name].controller.use_cfg = use_cfg

    # model.unet = torch.compile(model.unet)
# IMAGE_SIZE = 256

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 20,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    transform_coordinates=None,
    mask_obj = None,
    optimize_steps = 0.2,
    latent_replace = 0.2,
    lr = 0.0,
    optimize_embeddings=False,
    optimize_latents = False,
    ddim_latents = None,
    ddim_noise = None,
    edit_type = "geometry_editor",
    fast_start_steps = 0.0,
    num_first_optim_steps = 5,
    use_adaptive_optimization = True,
    adain_latents_steps=0.95,
):

    global_loss_log_dict = {}

    global GUIDANCE_SCALE, SKIP_OPTIM_STEPS, PROGRESS_BAR

    skip_optim_steps = SKIP_OPTIM_STEPS
    print("Using guidance scale: ", GUIDANCE_SCALE, " skip optim steps: ", SKIP_OPTIM_STEPS)
    batch_size = len(prompt)
    register_attention_control_diffusers(model, controller, transform_coordinates)
    model.uet = torch.compile(model.unet, mode = "reduce-overhead")
    
    height = width = IMAGE_SIZE
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [UNCOND_TEXT] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent[:1], model, height, width, generator, batch_size)

    if edit_type == "geometry_stitch":
        # Add latent for stitched object
        latents[1:2] = ddim_latents[-1][1:2]
    

    # print(latent.shape, latent.shape, ddim_latents[-1].shape)
    model.scheduler.set_timesteps(num_inference_steps)

    for param in model.unet.parameters():
        param.requires_grad = False

    context_save = None
    optimizer = None
    first_optim_complete = False
    l_eff = lr * skip_optim_steps * (50 / (NUM_DDIM_STEPS + 1e-8))
    scaler = None
    # scaler = torch.cuda.amp.GradScaler(growth_interval=1)
    pbar = tqdm(model.scheduler.timesteps[-start_time:])



    if transform_coordinates is not None:
        t_coords_m = reshape_transform_coords(transform_coordinates, in_mat_shape=controller.image_mask.shape).tile(controller.image_mask.shape[0], 1, 1, 1).type_as(text_embeddings)
        controller.mask_new_warped = binarize_tensor(warp_grid_edit(controller.image_mask[:, None], t_coords_m, padding_mode='zeros', align_corners=True, mode=MODE)).type_as(text_embeddings)
    
    for i, t in enumerate(pbar):

        # SPLATTER.clear_cache()
        SPLATTER.radius = max(1, int(SPLATTER.radius * 0.95))
        SPLATTER.tau = max(SPLATTER.tau * 0.8, 1e-1)
        SPLATTER.points_per_pixel = max(int(SPLATTER.points_per_pixel * 0.9), 4)
        # print(i)

        alpha_t = model.scheduler.timesteps[i]
        # print(alpha_t, i, t)
        alpha_t = model.scheduler.alphas_cumprod[alpha_t]
        beta_t = 1 - alpha_t
        # print(alpha_t, i, t)

        if uncond_embeddings_ is None:
        #             0, 1 are uncond embeddings and 2, 3 are text embeddings
        # edit only embedding 1
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
            
        
        if type(controller).__name__.startswith("AttentionGeometry"):
            # del controller.loss
            # controller.loss = 0.0

            clear_controller_loss(controller)
            # print(i, (fast_start_steps * len(model.scheduler.timesteps[-start_time:])), (len(model.scheduler.timesteps[-start_time:])), (fast_start_steps)) 

            
            if ((i < optimize_steps * len(model.scheduler.timesteps[-start_time:])) and (i % skip_optim_steps ==0) and (i >= (fast_start_steps * len(model.scheduler.timesteps[-start_time:])))):
                
                

                
                if not first_optim_complete and (fast_start_steps > 0.0):
                    num_optim_steps = num_first_optim_steps
                    first_optim_complete = True
                else:
                    num_optim_steps = 1


                best_loss = 1e+8
                best_latents = None
                best_context = None

                # if edit_type == "geometry_stitch":
                #     l_eff = l_eff * 0.98
                # # elif edit_type == "geometry_remover":
                # #     l_eff = 50 * lr * beta_t * skip_optim_steps * (50 / (NUM_DDIM_STEPS + 1e-8))
                # else:

                # if edit_type == "geometry_editor":
                #     l_eff = 50 * lr * beta_t * skip_optim_steps * (50 / (NUM_DDIM_STEPS + 1e-8))

                # else:
                l_eff = lr * (50 - i) * skip_optim_steps * (50 / (NUM_DDIM_STEPS + 1e-8))


                if edit_type == "geometry_stitch":
                    set_attn_processor_for_edit(model, coords_base=(0, 2), coords_edit=(2, 3), use_cfg=False)
                else:
                    set_attn_processor_for_edit(model, coords_base=(0, 1), coords_edit=(1, 2), use_cfg=False)
                    

                transform_coordinates = transform_coordinates.detach()
                # model.unet.zero_grad()
                latents_in = latents.detach().float().requires_grad_(True)
                orig_norm = norm_tensor(latents_in[-1:].detach()).item()

                if context_save is None:
                    context_in = context.detach().float().requires_grad_(True)
                else:
                    context_in = context_save.detach().float().requires_grad_(True)


                if optimizer is None:
                    optimizer = torch.optim.SGD([latents_in, context_in], lr = l_eff, momentum=0.9)
                else:
                    optimizer.param_groups[0]["params"] = [latents_in, context_in]
                optimizer.zero_grad(set_to_none=True)

                for opt_iter in range(num_optim_steps):

                    with torch.enable_grad():
                        for param in model.unet.parameters():
                            param.requires_grad = False


                        # model.unet = model.unet.eval()
                        # with torch.autocast(device_type="cuda", dtype=torch.float16):
                        # if 1:
                            # with torch.profiler.profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
                                
                            # with torch.autograd.profiler.profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:
                        if edit_type == "geometry_stitch":
                            latent_opt, noise_opt = diffusion_step(model, controller, latents_in, context_in[3:], t, guidance_scale, low_resource=False, transform_coords=transform_coordinates, use_cfg=False, return_noise=True)
                        else:
                            latent_opt, noise_opt = diffusion_step(model, controller, latents_in, context_in[2:], t, guidance_scale, low_resource=False, transform_coords=transform_coordinates, use_cfg=False, return_noise=True)

                        # model.unet.zero_grad()
                        # sds_grad = torch.sum((noise_opt[-1:] - noise_opt[:1]).detach() * latents_in[-1:]) + 0 * torch.sum(context_in)
                        controller.loss = controller.loss#.detach() * 0.0 + 1.0 * sds_grad
                        if controller.loss.detach().item() < best_loss:
                            best_latents = latents_in
                            best_context = context_in
                            best_loss = controller.loss.detach().item()


                        # print(opt_iter, " ", best_loss)
                        pbar.set_description_str(desc = "Loss: %f" % best_loss)

                        if PROGRESS_BAR is not None:
                            PROGRESS_BAR(i / NUM_DDIM_STEPS, desc="Editing Optimization Loss: %f" % best_loss)

                        if edit_type == "geometry_stitch":
                            latents_in, context_new = _update_latent(latents_in, controller.loss, l_eff, controller.mask_new_warped[:1], context_in, scaler = scaler, optimizer = optimizer)
                        else:
                            latents_in, context_new = _update_latent(latents_in, controller.loss, l_eff, None, context_in, scaler = scaler, optimizer = optimizer)

                        
                        if num_optim_steps == 1:
                            best_latents = latents_in
                            best_context = context_new
                        
                        context_in = context_new.detach().float().requires_grad_(True)
                        latents_in = latents_in.detach().float().requires_grad_(True)


                        out_loss_log_dict = convert_loss_log_to_numpy(controller.loss_log_dict)
                        
                        if use_adaptive_optimization:
                            print("[INFO]: Using Adaptive Optimization")
                            if edit_type == "geometry_editor":
                                adaptive_optimization_step_editing(controller, i, skip_optim_steps, out_loss_log_dict)
                            elif edit_type == "geometry_stitch":
                                adaptive_optimization_step_stitching(controller, i, skip_optim_steps, out_loss_log_dict)
                            elif edit_type == "geometry_remover":
                                adaptive_optimization_step_remover(controller, i, skip_optim_steps, out_loss_log_dict)
                        else:
                            print("[INFO]: Not Using Adaptive Optimization Results may not be optimal")


                                # pass 
                        # print(out_loss_log_dict)
                        global_loss_log_dict[i] = out_loss_log_dict
                        print(out_loss_log_dict["self"])
                        print(out_loss_log_dict["cross"])

                        # del controller.loss
                        # controller.loss = 0.0
                        clear_controller_loss(controller)
                        controller.cur_step -= 1

                        # latents_in = _update_latent(latents_in, controller.loss, lr * (50 - i))
                    
                    # if optimize_latents and (i < 0.6 * optimize_steps * len(model.scheduler.timesteps[-start_time:])) :
                if optimize_latents:
                    # print("Updating latents")
                    latents = best_latents.detach()
                    # Preserving norm
                    latents[-1:] = latents[-1:] * orig_norm / (norm_tensor(latents[-1:].detach()).item())
                    
                
                if best_context is not None and optimize_embeddings:
                    # print("Updating context embeddings")
                    context = best_context.detach()
                    context_save = context
                    # print(context.shape)

                pbar.set_description_str(desc = "Loss: %f" % best_loss)
                
                if PROGRESS_BAR is not None:
                    PROGRESS_BAR(i / NUM_DDIM_STEPS, desc="Editing Optimization Loss: %f" % best_loss)

                pbar.refresh()
                # pbar.set_postfix({'Loss': controller.loss.detach().item()})
            
                del latents_in
                del context_new
                del best_context
                del best_latents

                    
                with torch.no_grad():
                    if edit_type == "geometry_stitch":
                        set_attn_processor_for_edit(model, coords_base=(3, 5), coords_edit=(5, 6), use_cfg=True)
                    else:
                        set_attn_processor_for_edit(model, coords_base=(2, 3), coords_edit=(3, 4), use_cfg=True)

                    # with torch.no_grad():
                    #     latents[-1:] = adain(latents[-1:], latents[:1], dim = 1).detach()
                    
                    # with torch.no_grad():
                    #     latents[-1:] = adain_latents(latents[-1:], latents[:1]).detach()

                    latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, transform_coords=transform_coordinates)
                    
                
            elif (i < fast_start_steps * len(model.scheduler.timesteps[-start_time:])):
                pass
            else:

                if context_save is not None:
                    context = context_save
                # set_attn_processor_for_edit(model, coords_base=(2, 3), coords_edit=(3, 4), use_cfg=True, perform_edit=False)
                
                with torch.no_grad():
                    if edit_type == "geometry_stitch":
                        set_attn_processor_for_edit(model, coords_base=(3, 5), coords_edit=(5, 6), use_cfg=True)
                    else:
                        set_attn_processor_for_edit(model, coords_base=(2, 3), coords_edit=(3, 4), use_cfg=True)

                    latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, transform_coords=transform_coordinates)

                    # if (i < adain_latents_steps * len(model.scheduler.timesteps[-start_time:]) and i > optimize_steps * len(model.scheduler.timesteps[-start_time:])):
                    #     print("adain")
                    #     with torch.no_grad():
                    #         latents[-1:] = adain_latents(latents[-1:], latents[:1]).detach()
            
            if ddim_latents is not None:
                i_n = len(ddim_latents) - 2 - i
                latents = torch.cat([ddim_latents[i_n].type_as(latents.detach()), latents[-1:].detach()], 0)
                
            if PROGRESS_BAR is not None:
                PROGRESS_BAR(i / NUM_DDIM_STEPS, desc="Optimizing Latents")

            if not type(controller).__name__ == "AttentionGeometryRemover" and (not type(controller).__name__ == "AttentionGeometryStitchSingle") :
            
                with torch.no_grad():
                    # print("latent warp")
                    if (i < len(model.scheduler.timesteps[-start_time:]) * latent_replace and mask_obj is not None) or (i < len(model.scheduler.timesteps[-start_time:]) * fast_start_steps):
                        t_coords = reshape_transform_coords(transform_coordinates, in_mat_shape=latents[1:].shape).type_as(latents)
                        i_mask_transform = controller.mask_new_warped[:1].detach()
                        i_mask = (T.Resize(size=(latents.shape[-1], latents.shape[-1]))(i_mask_transform) > 0.5) * 1.0
                        i_mask = i_mask.type_as(latents)

                        
                        # if edit_type == "geometry_stitch":

                        if i < len(model.scheduler.timesteps[-start_time:]) * fast_start_steps:
                            latents[-1:] = latents[:1] * (1 - i_mask) + i_mask * warp_grid_edit(latents[-2:-1].detach().clone(), t_coords, padding_mode='zeros', align_corners=True, mode=MODE)
                            
                        else:
                            latents[-1:] = latents[-1:] * (1 - i_mask) + i_mask * warp_grid_edit(latents[-2:-1].detach().clone(), t_coords, padding_mode='zeros', align_corners=True, mode=MODE)
                        # print(transform_coordinates)
                        # print("Warping latents")
                        # else:
                            # latents[1:] = latents[1:] * (1 - i_mask) + i_mask * warp_grid_edit(latents[:1].detach().clone(), t_coords, padding_mode='zeros', align_corners=True, mode=MODE)

        else:
            
            latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, transform_coords=transform_coordinates)
        

        # print(latents.shape)

        # exit()
    # print(latents[1], ddim_latents[0].shape)
    # print(ddim_latents[0].min(), ddim_latents[0].max())
    # print(latents[1].min(), latents[1].max())
    # latents[1] = (((latents[1] - latents[1].min()) / ((latents[1].max() - latents[1].min()) + 1e-8)) * (ddim_latents[0].max() - ddim_latents[0].min())) + ddim_latents[0].min()
    # print(latents[1].min(), latents[1].max())

    if return_type == 'image':
        image = latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent, global_loss_log_dict



def run_and_display(ldm_stable, prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, transform_coordinates=None, mask_obj = None, optimize_steps = 0.0, latent_replace = 0.0, lr = 0.0, optimize_embeddings = False, optimize_latents= True, ddim_latents = None, ddim_noise = None, edit_type = "geometry_editor", fast_start_steps = 0.0, num_first_optim_steps = 1, use_adaptive_optimization = True):

    images, x_t, global_loss_log_dict = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, transform_coordinates=transform_coordinates, mask_obj = mask_obj, optimize_steps = optimize_steps, latent_replace = latent_replace, lr = lr, optimize_embeddings = optimize_embeddings, optimize_latents = optimize_latents, ddim_latents = ddim_latents, ddim_noise = ddim_noise, edit_type = edit_type, fast_start_steps = fast_start_steps, num_first_optim_steps = num_first_optim_steps, use_adaptive_optimization = use_adaptive_optimization)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t, global_loss_log_dict

def show_image(img):
    plt.imshow(img)
    plt.show()


import importlib
from util import vis_utils
importlib.reload(vis_utils)
import os
import matplotlib.pyplot as plt
import torchvision




# @torch.autocast("cuda", dtype=torch.half)
@log_args(logging.getLogger())
def perform_geometric_edit(image, depth, image_mask, transform_in, prompt = "", ldm_stable_model = None, tokenizer_model = None, scheduler_in = None, 
    cross_replace_steps = {'default_': 0.95},
    self_replace_steps = 0.95,
    optimize_steps = 0.6,
    lr = 0.03,
    latent_replace = 0.6,
    optimize_embeddings = True,
    optimize_latents = True,
    obj_edit_step = 1.0,
    perform_inversion = True,
    guidance_scale = 7.5,
    skip_optim_steps = 1,
    num_ddim_steps = 50,
    splatting_radius = 1.3,
    edit_type = "geometry_editor",
    image_stitch = None,
    progress = None,
    fast_start_steps = 0.0,
    num_first_optim_steps = 1,
    loss_weights_dict = None,
    return_loss_log_dict = False,
    splatting_tau = 1.0,
    splatting_points_per_pixel = 15,
    use_adaptive_optimization = True,
    return_attention_maps = False,
    unet_path = "",
    ):
    torch.cuda.empty_cache()

    global SEED

    

    print("[INFO]: Using prompt: \"", prompt, "\" ")
    print("[INFO]: torch state: ", torch.get_rng_state())

    rng_state = torch.get_rng_state()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(SEED)
    torch.cuda.manual_seed_all(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.manual_seed(SEED)


    max_opt = max(self_replace_steps, cross_replace_steps['default_'])
    if optimize_steps > max_opt:
        optimize_steps = max_opt

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # torch.autograd.detect_anomaly(check_nan=False)
    # torch.autograd.profiler.profile(enabled=False)
    torch.backends.cudnn.benchmark = True
    # torch.use_deterministic_algorithms(True)

    SPLATTER.clear_cache()
    SPLATTER.radius = splatting_radius
    SPLATTER.tau = splatting_tau
    SPLATTER.points_per_pixel = splatting_points_per_pixel

    global TOKENIZER, LDM_STABLE, SCHEDULER, PROGRESS_BAR
    global tokenizer, ldm_stable, scheduler

    PROGRESS_BAR = progress
    global GUIDANCE_SCALE, SKIP_OPTIM_STEPS, NUM_DDIM_STEPS
    
    GUIDANCE_SCALE = guidance_scale
    SKIP_OPTIM_STEPS = skip_optim_steps
    NUM_DDIM_STEPS = num_ddim_steps




    image_mask = torch.tensor(image_mask).float()
    # d_path = "/home/ec2-user/SageMaker/test_sd/DPT/outputs/debug_outputs_4/"
    # transforms_path = d_path + "transform_coordinates.npy"
    t_coords_depth, p_image, projected_image_amodal_mask = vis_utils.get_transform_coordinates(image / 255.0, depth, image_mask, transform_in = transform_in, return_mesh=True)



    if edit_type == "geometry_stitch" or edit_type == "geometry_stitch_single":
        if image_stitch is None:
            raise Exception("[ERROR]: No image provided for Stitching")


        t_coords_depth = torch.tensor(t_coords_depth)
        image_warped = warp_grid_edit((torch.tensor(image_stitch[None]).permute(0, -1, 1, 2) / 255.0).float(), t_coords_depth[None].float(), padding_mode='zeros', align_corners=True, mode=MODE)

        p_image = (image_warped[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")

        print(image_mask.shape, image_mask.max(), image.shape, image.max())
        mask_warp = warp_grid_edit((torch.tensor(image_mask[None, None])).float(), t_coords_depth[None].float(), padding_mode='zeros', align_corners=True, mode=MODE)

        mask_warped = (mask_warp[0, 0].detach().cpu().numpy() > 0.5)[..., None] * 1.0

        stitched_image = (p_image * mask_warped + image * (1.0 - mask_warped)).astype("uint8")
        # plt.imsave("./test.png", stitched_image)
        # exit()

        if edit_type == "geometry_stitch_single":
            image = stitched_image
        # else:
        #     print("Setting depth to 0.5 and using stitched image as image_stitch")
        #     image_stitch = stitched_image
        #     image_mask = torch.tensor(mask_warped[..., 0]).type_as(image_mask)
        #     transform_in = torch.eye(4).type_as(transform_in)
        #     # As image is stitched we set transform to I

        #     t_coords_depth, p_image = vis_utils.get_transform_coordinates(image / 255.0, 0.5 * np.ones_like(depth), image_mask, transform_in = transform_in)
        #     t_coords_depth = torch.tensor(t_coords_depth)
        #     # print(image_mask.shape, image_mask.dtype, image_mask.max(), image_mask.min())
        #     # print(image_mask.shape, image_mask.dtype, image_mask.max(), image_mask.min())
            

    transform_coordinates = t_coords_depth[None]
    transform_coordinates=torch.tensor(transform_coordinates).detach()

        
        


    ldm_stable = LDM_STABLE
    tokenizer = TOKENIZER
    scheduler = SCHEDULER

    if scheduler_in is not None and (unet_path == "" or unet_path == UNET_NAME):
        print("using pre loaded model.")
        ldm_stable = ldm_stable_model
        tokenizer = tokenizer_model
        scheduler = scheduler_in
        

    elif (ldm_stable is None) or (tokenizer is None) or (scheduler is None):
        ldm_stable, tokenizer, scheduler = load_model(unet_path)

    else:
        print("No model loading required")

    LDM_STABLE = ldm_stable
    TOKENIZER = tokenizer
    SCHEDULER = scheduler    

    null_inversion = NullInversion(ldm_stable)

    state = torch.get_rng_state()

    if edit_type == "geometry_stitch_single":
        image_stitch = None


    (image_gt, image_enc), x_t, uncond_embeddings, ddim_latents, ddim_noise = null_inversion.invert(image, prompt, offsets=(0,0,0,0), verbose=True, perform_inversion = perform_inversion, image_2 = image_stitch)
    print("inversion done!")





    # prompts = ["", "road"]
    if edit_type == "geometry_stitch":
        prompts = [prompt, prompt, prompt]
        if image_stitch is None:
            raise Exception("[ERROR]: ]No image provided for Stitching")
    else:
        prompts = [prompt, prompt]

    image_mask_torch = image_mask
    m_i_1 = image_mask_torch[None, None]
    # Params


    if edit_type == "geometry_editor":
        controller = AttentionGeometryEdit(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=None, local_blend=None, controller=None, image_mask=image_mask.detach().cpu().numpy(), empty_scale = 0.0, use_all = False, obj_edit_step = obj_edit_step)
    elif edit_type == "geometry_remover":
        controller = AttentionGeometryRemover(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=None, local_blend=None, controller=None, image_mask=image_mask.detach().cpu().numpy(), empty_scale = 0.0, use_all = False, obj_edit_step = obj_edit_step)
    
    elif edit_type == "geometry_stitch":
        controller = AttentionGeometryStitch(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=None, local_blend=None, controller=None, image_mask=image_mask.detach().cpu().numpy(), empty_scale = 0.0, use_all = False, obj_edit_step = obj_edit_step)
    elif edit_type == "geometry_stitch_single":
        controller = AttentionGeometryStitchSingle(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=None, local_blend=None, controller=None, image_mask=image_mask.detach().cpu().numpy(), empty_scale = 0.0, use_all = False, obj_edit_step = obj_edit_step)
    else:
        raise NotImplementedError

    if return_attention_maps:
        print("[INFO]: Storing Attention Maps")
        controller.store_attention_maps = True

    
    print("[INFO]: Setting Amodal Mask", projected_image_amodal_mask.shape)
    
    controller.amodal_mask = torch_erode(torch.tensor(projected_image_amodal_mask))


    if loss_weights_dict is not None:
        controller.loss_weight_dict = loss_weights_dict
        controller.default_loss_weights = loss_weights_dict
        print("[INFO]: Found loss weights dictionary for: ", edit_type)
    # elif edit_type == "geometry_stitch":
    #     lwd = {"self": {"sim": sim_loss_w_self, "scale": scale_loss_w_self}, "cross": { "sim": sim_loss_w_cross, "scale": scale_loss_w_cross}}
    #     controller.loss_weight_dict = lwd


    torch.set_rng_state(state)

    print("performing edit")

    generator = torch.Generator("cuda")
    generator.manual_seed(SEED)
    images, _, global_loss_log_dict = run_and_display(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, 
    transform_coordinates=transform_coordinates, mask_obj = m_i_1, optimize_steps=optimize_steps, latent_replace=latent_replace, lr = lr, optimize_embeddings = optimize_embeddings, optimize_latents=optimize_latents, ddim_latents=ddim_latents, verbose=False, ddim_noise = ddim_noise, edit_type = edit_type, fast_start_steps = fast_start_steps, num_first_optim_steps = num_first_optim_steps, use_adaptive_optimization = use_adaptive_optimization)


    # print(len(images))
    # print(images[1].shape, image.shape, image.max(), image.min(), images[1].max(), images[1].min())


    edited_image = images[-1]
    if edit_type == "geometry_editor":
        
        t_coords_depth = torch.tensor(t_coords_depth)[None]
        image_warped = warp_grid_edit((torch.tensor(image[None]).permute(0, -1, 1, 2) / 255.0).float(), t_coords_depth.float(), padding_mode='zeros', align_corners=True, mode=MODE)
        
        
        p_image = (image_warped[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")


        # print(p_image.min(), p_image.max(), p_image.shape)
        # p_image_m = (p_image).astype("uint8")
        mask_edit = controller.mask_new_warped[0, 0].detach().cpu().numpy()
        # print(mask_edit.shape, image_mask.shape, mask_edit.max(), image_mask.max())
        mask_im = image_mask.detach().cpu().numpy()
        mask_changed = ((mask_edit + mask_im) > 0.5) * 1.0
        mask_wo_edit = ((np.ones_like(mask_changed) - (mask_changed)) > 0.5) * 1.0
        p_image_new = (mask_wo_edit[..., None] * image + mask_edit[..., None] * p_image).astype("uint8")
        mask_source = ((mask_edit + mask_wo_edit) > 0.5) * 1.0
        mask_template = ((mask_im + mask_wo_edit) > 0.5) * 1.0
        # edited_image = match_histograms(edited_image, image, channel_axis=-1)
        edited_image = masked_histogram_matching(edited_image, p_image_new, mask_source, mask_source)
        # plt.imsave("./test.png", p_image_new)
    elif edit_type == "geometry_stitch":
        mask_edit = controller.mask_new_warped[0, 0].detach().cpu().numpy()
        edited_image = masked_histogram_matching(edited_image, image, 1.0 - mask_edit)
    elif edit_type == "geometry_remover":
        edited_image = masked_histogram_matching(edited_image, image, 1.0 - image_mask)

    images[-1] = edited_image
    print("edit done")

    ldm_stable.unet.set_attn_processor(VanillaAttentionProcessor())
    
    del null_inversion
    # torch.set_rng_state(rng_state)
    if return_loss_log_dict:
        
        if return_attention_maps:
            return images, global_loss_log_dict, controller.attention_store
        return images, global_loss_log_dict

    if return_attention_maps:
        return images, controller.attention_store
    return images




if __name__=="__main__":



    # from util.vis_utils import read_depth, get_transform_coordinates



    # d_path = "/home/ec2-user/SageMaker/test_sd/DPT/outputs/debug_outputs_2/"
    d_path = "/home/ec2-user/SageMaker/data/epfl-cars/epfl-gims08/tripod-seq/"
    d_seg_path = "/home/ec2-user/SageMaker/data/epfl_segmented/"
    d_depth_path = "/home/ec2-user/SageMaker/data/epfl-depths/"
    d_path = os.path.join(d_path, "")
    d_seg_path = os.path.join(d_seg_path, "")
    d_depth_path = os.path.join(d_depth_path, "")


    img_name = "tripod_seq_08_024.jpg"

    image_path = d_path + img_name
    segmentation_path = d_seg_path + "mask_" + img_name #"mask_tripod_seq_04_045.jpg"
    depth_path = d_depth_path + img_name.split(".")[0] + ".pfm"
    depth = vis_utils.read_depth(depth_path, (IMAGE_SIZE, IMAGE_SIZE))

    image_mask = (load_256(segmentation_path)[..., 0] / 255.0 > 0.5) * 1.0


    image = load_256(image_path)
    transforms_path = d_path + "transform_coordinates.npy"
    # prompt = "a car in the center of new york"
    # prompt = "an object in a scene"
    prompt = ""

    transform_in = vis_utils.rotateAxis(10, 1)


    images = perform_geometric_edit(image, depth, image_mask, transform_in, prompt)

    # print(images, len(images))
    output_dir = "./output_code/"

    output_dir = fix_dir_path(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    plt.imsave(output_dir + "output_nt.png", images[1])
    plt.imsave(output_dir + "input_nt.png", images[0])


    # exit()


    
