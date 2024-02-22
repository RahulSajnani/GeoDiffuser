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
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import math


from omegaconf import OmegaConf

from run_generation import get_argument_parser, preprocess_image, instantiate_from_config, load_checkpoint, sample_model
from PIL import Image

from transformers import AutoImageProcessor, DetrModel, DetrForObjectDetection

TRANSLATE_MODEL = None
ROTATE_MODEL = None
DETR_MODEL = None

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

    global TRANSLATE_MODEL, ROTATE_MODEL

    model = None
    if task == "translate":
        model = TRANSLATE_MODEL
    elif task == "rotate":
        model = ROTATE_MODEL

    if model is None:
        print("LOADING MODEL!")
        config = OmegaConf.load(f"configs/sd-objaverse-{task}.yaml")
        OmegaConf.update(config,"model.params.cond_stage_config.params.device",device)
        model = instantiate_from_config(config.model)
        # model.cpu()
        model.to(device)
        load_checkpoint(model, checkpoint_path)
        model.eval()
        print("FINISHED LOADING!")
        if task == "translate":
            TRANSLATE_MODEL = model
        elif task == "rotate":
            ROTATE_MODEL = model
    else:
        print("using pre-loaded model")


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


def process_depth(depth):
    depth = depth.max() - depth
    depth = depth / (depth.max() + 1e-8)
    depth[depth > 0.9] = 1000000.0
    mask = (depth < 100.0) * 1.0

    return depth, mask


def normalize_vector(v, eps = 1e-12):
    return v / (np.linalg.norm(v) + eps)

def get_azimuth_angle(r):

    t = normalize_vector(np.array([1, 0, 1]).astype("float32"))

    rotated_t = r @ t
    # Projecting to y = 0 plane
    rotated_t[1] = 0
    rotated_t = normalize_vector(rotated_t)

    # print(t, rotated_t)
    
    angle = np.arccos(np.dot(rotated_t, t))

    return np.rad2deg(angle)

def get_translation(transform_mat, mask):
    
    mask = (((mask / 255.0) > 0.5) * 1.0)[..., 0]
    # print(mask.shape, mask.min(), mask.max())
    h, w = np.indices(mask.shape)


    w_obj_location = np.mean(w[mask > 0.5])
    h_obj_location = np.mean(h[mask > 0.5])

    x = float(w_obj_location / mask.shape[1])
    y = 1.0 - float(h_obj_location / mask.shape[0])
    return x, y


def get_task_from_transform(transform_mat, mask):


    assert mask is not None, "No transformed mask found, run ui_utils"
    r = transform_mat[:3, :3]
    rt = transform_mat[:3, -1]
    # print(r)
    dist_r = np.sum(np.abs(r - np.eye(3))) 
    dist_t = np.sum(np.abs(rt))

    # print(dist_r, dist_t)
    if np.linalg.det(r) < 0:
        # When reflection is present
        return "nothing", None
    x_out, y_out, z_out = 0, 0, 0
    # print(dist)
    if dist_r < 1e-8 and dist_t > 1e-4:
        # Only translation task
        task = "translate"
        x, y = get_translation(transform_mat, mask)

        return task, (x, y)

    elif dist_t <= 1e-4 and dist_r >= 1e-8:
        # Only rotation
        task = "rotate"
        angle = get_azimuth_angle(r)

        return task, angle

    elif (dist_t < 1e-8) and (dist_r < 1e-4):
        # Do nothing
        task = "nothing"
        return task, None

    else:
        # rotate and translate
        task = "both"
        x, y = get_translation(transform_mat, mask)
        angle = get_azimuth_angle(r)

        # If angle is small ignore
        if np.abs(angle) < 2:
            return "translate", (x, y)
        return task, (angle, x, y)





    # elif (np.sum(np.abs(rt)) > 1e-8) and (not np.all(depth == 0.5)):
        

    #     mask_p = (((mask[..., 0] / 255.0) > 0.5) * 1.0)
    #     depth_p, d_mask = process_depth(depth)
    #     # print()
    #     d_mean = np.mean(depth_p[(mask_p * d_mask) > 0.5])
    #     # d_mean = np.sum(process_depth(depth) * mask_p) / np.sum(mask_p)
    #     # print(d_mean)

    #     # Use depth map before and after projection to get the true angle
    #     rt[-1] += d_mean
    #     t = np.array([0, 0, d_mean]).astype("float32")
    #     # norm_t = np.linalg.norm(rt)
    #     # print(t, rt)
    #     x, y, z = cartesian_to_spherical(t[None])
    #     # print(rt[None].shape)
    #     xr, yr, zr = cartesian_to_spherical(rt[None, :])

    #     x_out += xr - x
    #     y_out += yr - y
    #     z_out += zr - z


    #     return x_out, y_out, z_out

    return 

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
    

def get_detr_model():
    
    global DETR_MODEL
    if DETR_MODEL is None:
        image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        # model = DetrModel.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        DETR_MODEL = model, image_processor
    else:
        model, image_processor = DETR_MODEL
    return model, image_processor

def normalize_image(im):

    if im.max() > 1:
        return im / 255.0
    else:
        return im

def add_margin(pil_img, color, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result

def get_input_for_detr(input_im, mask_im):

    # print(input_im.shape, mask_im.shape)
    # print(input_im.max(), mask_im.max())

    if len(mask_im.shape) > 2:
        mask_im = mask_im[..., 0]

    mask_im = (normalize_image(mask_im) > 0.5) * 1.0
    # input_im = normalize_image(input_im)

    h, w = np.indices(mask_im.shape)

    h_min, h_max = h[mask_im > 0.5].min(), h[mask_im > 0.5].max()
    w_min, w_max = w[mask_im > 0.5].min(), w[mask_im > 0.5].max()


    im_out = input_im[h_min:h_max, w_min:w_max]
    im_mask_out = mask_im[h_min:h_max, w_min:w_max]

    im_out = Image.fromarray(im_out)
    im_out.thumbnail([200, 200], Image.Resampling.LANCZOS)
    im_out = add_margin(im_out, (255, 255, 255), size=256)
    im_out = np.array(im_out)

    # input_im * mask_im[..., None]

    return im_out, im_mask_out


def get_object_category(im, mask, d_model, image_processor, threshold = 0.9):

    im, im_mask = get_input_for_detr(im, mask)
    image = Image.fromarray(im)
    target_sizes = torch.tensor([image.size[::-1]])
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = d_model(**inputs)


    
    results = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]   

    if len(results):
        return d_model.config.id2label[results["labels"][0].item()]
    else:
        # Failed identification
        return None


def write_string_to_file(f_path, text):

    f = open(f_path, "w")
    f.write(text)
    f.close()


def list_exp_details(exp_dict):

    for k in exp_dict.keys():
        if exp_dict[k] is None:
            print(k, " None")
        elif k != "path_name":
            print(k, " ", exp_dict[k].shape, " ", exp_dict[k].min(), " ", exp_dict[k].max())
        elif k == "path_name":
            print(k, " ", exp_dict[k])

def evaluate_geodiff(exp_root_folder, weights_folder):

    folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    folder_list.sort()

    d_model, image_processor = get_detr_model()

    num_k = 0

    for exp_folder in folder_list:
        
        num_k +=1
        print("Performing edit on: ", exp_folder)
        
        
        # continue
        exp_folder = complete_path(exp_folder)
        exp_dict = read_exp(exp_folder)
        # list_exp_details(exp_dict)


        im = exp_dict["input_image_png"]
        im_mask = exp_dict["input_mask_png"]
        transformed_mask = exp_dict["transformed_mask_square_png"]
        transform_mat = exp_dict["transform_npy"]
        im_size = exp_dict["image_shape_npy"]


        # get task
        task, params = get_task_from_transform(transform_mat, transformed_mask)
        # continue
        # get category
        category = get_object_category(im, im_mask, d_model, image_processor)


        print(task, params)
        print("detected category: ", category)
        if category is None:
            continue
        if task == "both":
            a, x, y = params
            task_1 = "rotate"
            ckpt = weights_folder + task_1 + ".ckpt"
            input_im, output_ims = edit(im, task_1, category, ckpt, rotation_angle=a)
            im_2 = np.array(output_ims[0])
            # plt.imsave("./geodiff_gen_1.png", resize_image(np.array(output_ims[0]), im_size))
            im_2 = im
            task_2 = "translate"
            ckpt = weights_folder + task_2 + ".ckpt"
            input_im, output_ims = edit(im_2, task_2, category, ckpt, x = x, y=y)
            gen_output = output_ims[0]

            # plt.imsave("./geodiff_gen_2.png", resize_image(np.array(output_ims[0]), im_size))

        elif task != "nothing":
            ckpt = weights_folder + task + ".ckpt"
            if task == "rotate":
                input_im, output_ims = edit(im, task, category, ckpt, a = params)
            else:
                input_im, output_ims = edit(im, task, category, ckpt, x = params[0], y = params[1])

            gen_output = output_ims[0]
            # plt.imsave("./geodiff_gen.png", resize_image(np.array(output_ims[0]), im_size))
        else:
            # Do nothing save input as generated
            gen_output = im
            # plt.imsave("./geodiff_gen.png", resize_image(np.array(input_im), im_size))


        out_path_dir = exp_folder + "object_edit/"
        os.makedirs(out_path_dir, exist_ok = True)
        

        write_string_to_file(out_path_dir + "category.txt", category)
        write_string_to_file(out_path_dir + "task.txt", task)
        if params is not None:
            np.save(out_path_dir + "params.npy", np.array(params))

        plt.imsave(out_path_dir + "result_object_edit..png", resize_image(np.array(gen_output), im_size))

        # if num_k == 2:
        #     exit()
        # exit()



    

if __name__ == "__main__":

    exp_root_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/editing/"
    weights_folder = "../../../weights/object-edit/"
    evaluate_geodiff(exp_root_folder, weights_folder)
    exit()
    # exp_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/editing/1"
    # exp_dict = read_exp(exp_folder)
    # im = exp_dict["input_image_png"]
    # im_mask = exp_dict["input_mask_png"]
    # transform_mat = exp_dict["transform_npy"]
    # depth = exp_dict["depth_npy"]
    # im_size = exp_dict["image_shape_npy"]
    # transformed_mask = exp_dict["transformed_mask_square_png"]

    # print(transform_mat)
    
    # task, params = get_task_from_transform(transform_mat, transformed_mask)

    # print(task, params)
    # # exit()

    # d_model, image_processor = get_detr_model()

    # category = get_object_category(im, im_mask, d_model, image_processor)

    # print("detected category: ", category)
    # if task == "both":
    #     a, x, y = params
    #     task_1 = "rotate"
    #     # ckpt = weights_folder + task_1 + ".ckpt"
    #     # input_im, output_ims = edit(im, task_1, category, ckpt, rotation_angle=a)
    #     # im_2 = np.array(output_ims[0])
    #     # plt.imsave("./geodiff_gen_1.png", resize_image(np.array(output_ims[0]), im_size))
    #     im_2 = im
    #     task_2 = "translate"
    #     ckpt = weights_folder + task_2 + ".ckpt"
    #     input_im, output_ims = edit(im_2, task_2, category, ckpt, x = x, y=y)
    #     plt.imsave("./geodiff_gen_2.png", resize_image(np.array(output_ims[0]), im_size))

    # elif task != "nothing":
    #     ckpt = weights_folder + task + ".ckpt"
    #     if task == "rotate":
    #         input_im, output_ims = edit(im, task, category, ckpt, a = params)
    #     else:
    #         input_im, output_ims = edit(im, task, category, ckpt, x = params[0], y = params[1])

    #     input_im, output_ims = edit(im, task, category, ckpt, x = 0.6, y=0.2)
    #     plt.imsave("./geodiff_gen.png", resize_image(np.array(output_ims[0]), im_size))
    # else:
    #     # Do nothing save input as generated
    #     plt.imsave("./geodiff_gen.png", resize_image(np.array(input_im), im_size))
        
    # plt.imsave("./geodiff_in.png", resize_image(np.array(im), im_size))
    # exit()


    # pass