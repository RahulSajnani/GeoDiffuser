import numpy as np
import cv2, glob, argparse, os
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from PIL.ImageOps import exif_transpose
import PIL
from copy import deepcopy

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


def get_axis_points(scale = 1.0):

    points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3) * scale

    return points

# def project_points_to_3D(im, K, d):

def translateMatrixFromVector(v):

    translation_matrix = np.eye(4)
    # translation_matrix[0,3] += x
    # translation_matrix[1,3] += y
    # translation_matrix[2,3] += z
    translation_matrix[:3, 3] += v

    return translation_matrix


def project_points_to_3D(im, K, d):

    h, w = np.indices(im[..., 0].shape)
    cam_coords = np.stack([w, h, np.ones_like(h)], -1) # h, w, 3

    # print(cam_coords.shape)
    pts_3D = (np.linalg.inv(K)[None, None] @ cam_coords[..., None])[..., 0] * d[..., None] # h,, w, 3
    return pts_3D, cam_coords

def get_geodiff_translation(im, K, d_in, transform_mat, obj_mask):

    print(transform_mat)

    d, mask_d = prepare_depth_for_projection(d_in, obj_mask[..., 0] / 255.0)


    pts_3D, cam_coords = project_points_to_3D(im, K, d) # h, w, 3

    pts_3D_pcd = np.reshape(np.transpose(pts_3D, (-1, 0, 1)), (3, -1)).T
    obj_center = np.mean(pts_3D_pcd[np.reshape(mask_d, (-1)) >= 0.5, :], 0)

    pts_3D_hom = np.concatenate([pts_3D, np.ones_like(pts_3D[..., :1])], -1) # h, w, 4

    T_mean = translateMatrixFromVector(obj_center)

    pose_transform = T_mean @ transform_mat @ np.linalg.inv(T_mean)

    transformed_coords = (pose_transform[None, None] @ pts_3D_hom[..., None])[..., 0] # h, w, 4, 1




    projected_pts = (K[None, None] @ transformed_coords[:, :, :3, None])[..., 0]

    projected_pts[:, :, :2] = (projected_pts[:, :, :2] / projected_pts[..., -1:])

    deviation = np.mean((projected_pts[..., :2] - cam_coords[..., :2])[mask_d >= 0.5], 0)
    

    return deviation, projected_pts, mask_d, cam_coords


    # projected_pts = projected_pts[mask_d > 0.5]
    # print(projected_pts.shape, projec)





def draw_axis(img, R, t, K, points, shift = None):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)

    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    # axisPoints = axi
    # print(axisPoints[-1])

    if shift is not None:
        
        w_i, h_i = axisPoints[-1].ravel()
        # print(w_i)
        h_new = h_i / shift[0]
        w_new = w_i / shift[1]
        shift = np.array([w_new - w_i, h_new - h_i])
        # print(shift)
        axisPoints = axisPoints + shift[None]
    # print(axisPoints)
    axisPoints = axisPoints.astype("int")
    # axisPoints = np.concatenate(axisPoints[:, 1:], axisPoints[:, :1], -1)
    # axisPoints = 
    # print(axisPoints[-1].ravel(), axisPoints[1].ravel())
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    return img

def get_mask_center(mask):

    m = mask.copy()
    if m.max() > 1:
        m = m / 255.0
    if len(m.shape) > 2:
        m = m[..., 0]

    h_ind, w_ind = np.indices(m.shape)
    h_mean = h_ind[m > 0.5].mean()
    w_mean = w_ind[m > 0.5].mean()

    return w_mean, h_mean

def prepare_depth_for_projection(d, mask_in = None):

    depth = d.copy()
    if mask_in is not None:
        obj_mask = mask_in

        if len(mask_in.shape) > 2:
            obj_mask = obj_mask[:, :, 0]
        if obj_mask.max() > 1.0:
            obj_mask = obj_mask / 255.0
    else:
        obj_mask = None
        
    if np.sum(depth) == 0.5 * (depth.shape[0] * depth.shape[1]):
        depth = np.ones_like(depth) * 0.5

    else:
        # Normalize depth
        depth = depth / (depth.max() + 1e-8)

        depth[depth > 0.95] = 10.0

    mask = (depth < 10.0) * 1.0

    if obj_mask is not None:
        mask = obj_mask * mask


    return depth, mask

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

def camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Constructs camera matrix
    '''

    K = np.array([[ focal_x,      0,     c_x],
                  [     0,  focal_y,     c_y],
                  [     0,      0,       1]])

    return K


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


def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height,width,_ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length,int(length*height/width)), PIL.Image.BILINEAR)
    mask  = cv2.resize(mask, (length,int(length*height/width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask


def get_points_geodiff(img,
               sel_pix,
               target_pts):
    # collect the selected point
    # sel_pix.append(evt.index)

    # draw points
    points = []
    all_points = []

    for idx, point in enumerate(sel_pix):
        cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        cv2.circle(img, tuple(target_pts[idx]), 10, (0, 0, 255), -1)

        # if idx % 2 == 0:
        #     # draw a red circle at the handle point
        #     cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        # else:
        #     # draw a blue circle at the handle point
        #     cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        points.append(tuple(target_pts[idx]))

        all_points.append(tuple(point))
        all_points.append(tuple(target_pts[idx]))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    return img, np.array(all_points)
    # return img if isinstance(img, np.ndarray) else np.array(img)

def draw_points(image, points):
    img = image.copy()

    for idx, point in enumerate(points):   
        cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    return img