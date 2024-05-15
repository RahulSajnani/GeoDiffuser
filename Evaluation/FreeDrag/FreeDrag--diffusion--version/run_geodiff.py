from utils.geodiff_utils import *
import os
os.environ['HF_HOME'] = "/oscar/scratch/rsajnani/rsajnani/research/.cache/hf"
# from utils.ui_utils import clear_all, store_img, train_lora_interface, run_drag
np.random.seed(1234)


def get_box_corners_mask(pts):

    w_min, h_min = np.min(pts, axis = 0)
    w_max, h_max = np.max(pts, axis = 0)
    # w_max = np.max(pts[:, 0])

    # h_min = np.min(pts[:, 1])
    # h_max = np.max(pts[:, 1])

    return w_min, w_max, h_min, h_max

def get_editing_mask(source_pts, target_pts, im_size):

    src_coords = get_box_corners_mask(source_pts)
    target_coords = get_box_corners_mask(target_pts)

    w_min = min(src_coords[0], target_coords[0])
    w_max = max(src_coords[1], target_coords[1])

    h_min = min(src_coords[2], target_coords[2])
    h_max = max(src_coords[3], target_coords[3])


    m = np.zeros(im_size).astype(np.uint8)

    m[int(h_min):int(h_max), int(w_min):int(w_max), :] = 255

    return m 

def run_freedrag_single(exp_folder, dragon_diff_model):

    exp_dict = read_exp(exp_folder)
    image = exp_dict["input_image_png"]
    mask = exp_dict["input_mask_png"]
    depth = exp_dict["depth_npy"]
    transform_mat = exp_dict["transform_npy"]
    print(exp_dict["prompt_txt"])
    K = camera_matrix(550, 550, image.shape[1] / 2.0, image.shape[0] / 2.0)

    d_path = complete_path(exp_dict["path_name"]) + "free_drag/"
    create_folder(d_path)

    


    # im, K, d_in, transform_mat, obj_mask
    
    translation_exp, flow, mask_pts, source_pts = get_geodiff_translation(image, K, depth, transform_mat, mask)
    t_w, t_h = translation_exp

    img = {"image": image, "mask": mask}
    _, _, masked_image, _ = store_img(img)

    im_size = masked_image.shape

    source_pts = source_pts[mask_pts >= 0.5, :2]
    target_pts = flow[mask_pts >= 0.5, :2]

    mask_updated = get_editing_mask(source_pts, target_pts, im_size)

    img = {"image": image, "mask": mask_updated}
    _, _, masked_image, _ = store_img(img)

    
    # drag_pts = flow[mask_pts >= 0.5]

    idx_array = np.random.choice(source_pts.shape[0] - 1, 10)
    input_drag_img = get_points_geodiff(masked_image, source_pts[idx_array].astype(int), target_pts[idx_array].astype(int))


    
    # deviation = flow - source_pts

    # flow_inv = source_pts + flow


    plt.imsave(d_path + "input_free_drag.png", input_drag_img)

    # print(t_w, t_h, exp_dict["path_name"], drag_pts.shape)

    exit()

    w_mean, h_mean=get_mask_center(mask)

    selected_points = [(w_mean, h_mean), (w_mean + t_w, h_mean + t_h)]
    print(selected_points)

    prompt = exp_dict["prompt_txt"]
    if prompt is None:
        prompt = ""

    output_image = dragon_diff_model.run_move(original_image=image, mask=mask, mask_ref=None, prompt=prompt, resize_scale=1.0, w_edit=4, w_content=6, w_contrast=0.2, w_inpaint=5.0, seed=42, selected_points=selected_points, guidance_scale=4, energy_scale=0.5, max_resolution=768, SDE_strength = 0.4, ip_scale=0.1)

    # print(type(output_image))


    output_image[0] = resize_image(output_image[0], exp_dict["image_shape_npy"])
    print(output_image[0].shape)
    plt.imsave(d_path + "result_free_drag.png", output_image[0])
    # plt.imsave("./test.png", output_image[0])
    # exit()
    pass

def run_freedrag_geodiff(exp_root_folder, dragon_diff_model):

    folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    folder_list.sort()

    print(folder_list)

    if check_if_exp_root(exp_root_folder):
        root_folders_list = folder_list
        for f in root_folders_list:
            folder_list = glob.glob(complete_path(f) + "**/")
            folder_list.sort()

            exp_cat = f.split("/")[-2]
            if not (exp_cat == "Translation_2D"):
                continue

            for exp_folder in folder_list:
                run_freedrag_single(exp_folder, dragon_diff_model)
    
            # exit()
    else:
        for exp_folder in folder_list:
            run_freedrag_single(exp_folder, dragon_diff_model)



if __name__ == "__main__":

    print("Running")
    pretrained_model_path = "runwayml/stable-diffusion-v1-5"
    # dragon_diff_model = DragonModels(pretrained_model_path=pretrained_model_path)
    dragon_diff_model = None
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--exp_root',
        required=False, default = None)
    args = parser.parse_args()

    # run_freedrag_geodiff(args.exp_root, dragon_diff_model)
    run_freedrag_single(args.exp_root, dragon_diff_model)

    pass