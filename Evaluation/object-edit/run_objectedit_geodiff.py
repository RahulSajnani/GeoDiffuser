import torch, detectron2
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from ldm.models.diffusion.ddim import DDIMSampler

from einops import rearrange

setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import cv2, os
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


from omegaconf import OmegaConf

from run_generation import get_argument_parser, preprocess_image, instantiate_from_config, load_checkpoint, sample_model
from PIL import Image


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



def get_detectron_model():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

    predictor = DefaultPredictor(cfg)

    return predictor

def edit(image, task, object_prompt, checkpoint_path, x = -1, y = -1, rotation_angle = 0, device="cuda", ddim_steps=50, num_samples=1, cfg_scale = 3.0):

    print("LOADING MODEL!")
    config = OmegaConf.load(f"configs/sd-objaverse-{task}.yaml")
    OmegaConf.update(config,"model.params.cond_stage_config.params.device",device)
    model = instantiate_from_config(config.model)
    model.cpu()
    load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()
    print("FINISHED LOADING!")

    image = Image.fromarray(image)
    input_im = preprocess_image(image).to(device)
    
    if task == "rotate":
        prompt = f"rotate the {object_prompt}"
        azimuth = math.radians(rotation_angle)
        T = torch.tensor([np.pi / 2, math.sin(azimuth), math.cos(azimuth),0])
    elif task == "remove":
        prompt = f"remove the {object_prompt}"
        T = torch.tensor([0.,0.,0.,0.])
    elif task == "insert":
        prompt = f"insert the {object_prompt}"
        T = torch.tensor([0,x,y,0])
    elif task == "translate":
        prompt = f"move the {object_prompt}"
        T = torch.tensor([0,x,y,0])

    print(prompt, T)
    

    sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(
        model,
        input_im,
        prompt,
        T,
        sampler,
        ddim_steps,
        num_samples,
        scale=cfg_scale
    )
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    input_im = ((input_im + 1.0) / 2.0).cpu()[0]
    input_im = 255.0 * rearrange(input_im.numpy(), 'c h w -> h w c')
    input_im = Image.fromarray(input_im.astype(np.uint8))

    # os.makedirs(args.save_dir,exist_ok=True)

    # input_im.save(os.path.join(args.save_dir,"input_im.png"))
    # for i,img in enumerate(output_ims):
    #     img.save(os.path.join(args.save_dir,f"{i}.png"))

    return input_im, output_ims

def run_object_edit(input_img, ):


    pass


if __name__ == "__main__":


    exp_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/editing/1"
    exp_dict = read_exp(exp_folder)
    im = exp_dict["input_image_png"]
    im_mask = exp_dict["input_mask_png"]
    # im_out = get_input(im, im_mask)
    transform_mat = exp_dict["transform_npy"]
    depth = exp_dict["depth_npy"]


    
    ckpt = "../../../weights/object-edit/translate.ckpt"
    task = "translate"
    input_im, output_ims = edit(im, task, "cat", ckpt, x = 0.9, y=0.6)

    plt.imsave("./geodiff_in.png", np.array(input_im))
    plt.imsave("./geodiff_gen.png", np.array(output_ims[0]))


    pass