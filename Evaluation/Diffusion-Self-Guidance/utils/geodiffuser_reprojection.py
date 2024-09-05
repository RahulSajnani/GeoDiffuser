import numpy as np
import cv2
# import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from PIL import Image
import json
from mpl_toolkits.mplot3d import Axes3D as mpl_3D
# import pptk
import glob
import os, argparse
# from util.warp_utils import forward_warp, pose_2_matrix, forward_splatting_warp, forward_splatting_pytorch3d_warp
from PIL import Image
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from torch import nn
from utils import geodiffuser_softsplat as softsplat
from pytorch3d.structures import Pointclouds
from pytorch3d.structures import Meshes
import numbers, math
import torchvision.transforms as T



EPS = 1e-2

SEED = 1234
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MODE="bilinear"


from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing, TexturesVertex, TexturesUV, MeshRenderer, MeshRasterizer
from pytorch3d.renderer.points import rasterize_points
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments


import numpy as np

FLOAT_EPS = np.finfo(np.float16).eps


def show_image(image):

    plt.imshow(image)
    plt.show()


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

        rotation_mat = torch.tensor([[np.cos(radians), -np.sin(radians),           0,          0],
                                 [np.sin(radians),  np.cos(radians),           0,          0],
                                 [              0,                0,           1,          0],
                                 [              0,                0,           0,          1]])

    elif axis == 1: # y - axis

        rotation_mat = torch.tensor([[np.cos(radians),                0,  np.sin(radians),          0],
                                 [              0,                1,                0,          0],
                                 [-np.sin(radians),               0, np.cos(radians),          0],
                                 [              0,                0,                0,          1]])

    elif axis == 0: # x - axis


        rotation_mat = torch.tensor([[             1,                0,                0,          0],
                                [              0,  np.cos(radians), -np.sin(radians),          0],
                                [              0,  np.sin(radians),  np.cos(radians),          0], 
                                [              0,                0,                0,          1]])
    
    return rotation_mat

def translateMatrix(x, y, z):

    translation_matrix = torch.eye(4)
    translation_matrix[0,3] += x
    translation_matrix[1,3] += y
    translation_matrix[2,3] += z

    return translation_matrix#.type_as(torch.double)



def camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Constructs camera matrix
    '''

    K = np.array([[ focal_x,      0,     c_x],
                  [     0,  focal_y,     c_y],
                  [     0,      0,       1]])

    return K
# Points generator
def get_grid(x, y):
    '''
    Get index grid from image
    '''
    # coords = np.indices((x, y)).reshape(2, -1)
    # return np.vstack((coords, np.ones(coords.shape[1])))

    y_i, x_i = np.indices((x, y))
    coords = np.stack([x_i, y_i, np.ones_like(x_i)], axis = -1).reshape(x*y, 3)
    # coords = np.indices((x, y)).reshape(2, -1)
    # return np.vstack((coords, np.ones(coords.shape[1])))
    # print(coords)
    return coords.T

def depth_decode(depth_image):

    # depth_image = np.array(depth_image)
    # # first 16 bits (first 2 channels) are 16-bit depth
    # R is the 8 LSB and G are the others
    depth_image_16 = depth_image[:,:,[1, 0]]
    # B are 8-bit version
    depth_image_8 = depth_image[:,:,2]
    # plt.imshow(depth_image_8)
    # plt.show()

    # last 8 are empty
    depth_single_channel = np.zeros((depth_image_16.shape[0], depth_image_16.shape[1]))
    # convert 16 bit to actual depth values
    for i in range(depth_single_channel.shape[0]):
        for j in range(depth_single_channel.shape[1]):
            bit_str = '{0:08b}'.format(depth_image_16[i, j, 0]) + '{0:08b}'.format(depth_image_16[i, j, 1])
            depth_single_channel[i, j] = int(bit_str, 2)

    depth_single_channel /= 1000
    # print(np.min(depth_single_channel))

    depth_single_channel
    depth_vector = depth_single_channel.reshape(1, -1)
    
    return depth_single_channel, depth_vector 

def extract_depth_tiff(depth_tiff):

    '''
    Extract depth from tiff image
    '''

    depth = np.array(depth_tiff)

    depth = 10 * (1 - depth)
    depth = depth.reshape(1, -1)
    return depth

def depth_2_point_cloud(invK, image, depth_image, depth_tiff=None):
    '''
    Convert depth map to point cloud

    '''
    points_hom = get_grid(image.shape[0], image.shape[1])
    


    if depth_tiff != None:
        print('tiff\n')
        depth = extract_depth_tiff(depth_tiff)
    else:
        depth = extract_depth(depth_image)
    # depth_map, depth = depth_decode(depth_image)
    
    # print(np.min(depth), np.max(depth[depth<30]))
    point_3D = invK @ points_hom
    point_3D = point_3D / point_3D[2, :]
    point_3D = point_3D * depth
    
    return point_3D
    
def save_point_cloud(K, image, mask, depth, num, output_directory, depth_tiff = None, max_depth = np.inf):
    '''
    Save point cloud given depth map
    '''

    directory = '%s/' % output_directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_colors = image.reshape(-1, 3)
    # print(image_colors.shape)
    invK = np.linalg.inv(K)
    # invK[0,0] *= 1
    # print(invK)
    point_cloud = depth_2_point_cloud(invK, image, depth, depth_tiff)   
    # point_cloud[0, :] *= -1
    mask = mask.reshape(-1, 1)
    mask = mask > 0.5
    # print(mask.shape)
    image_colors = image_colors[mask[:, 0], :]
    point_cloud = point_cloud[:, mask[:, 0]]
    # image_colors = image_colors[point_cloud[2,:] < 30, :]
    # point_cloud = point_cloud[:, point_cloud[2,:] < 30]
    
    image_colors = image_colors[point_cloud[2,:] < max_depth, :]
    point_cloud = point_cloud[:, point_cloud[2,:] < max_depth]
    

    # image_colors = image_colors[point_cloud[2,:] < 300, :]
    # point_cloud = point_cloud[:, point_cloud[2,:] < 300]
    
    # point_cloud[, :] = -point_cloud[2,:]
    # point_cloud[2, :] *= 10
    # print(np.min(point_cloud[2, :]), np.max(point_cloud[2,:]))
    # print(point_cloud.shape)
    
    
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    # pcd.colors = o3d.utility.Vector3dVector(image_colors)
    # print(pcd.colors)
    # ply_name = "%s/frame_%06d_point_cloud.ply" % (output_directory, num)
    # o3d.io.write_point_cloud(ply_name, pcd)
    return point_cloud#, pcd

def extract_depth(depth_map):

    '''
    Get a depth vector from image
    '''

    depth = depth_map.reshape(1, -1)
    # depth = depth.max() - depth
    # depth = 30 - 30 * depth
    # depth = 1 / (depth + 0.001)
    
    return depth

def quat2mat(quat):
    
    x, y, z, w = quat
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def json_to_numpy(pose):
    pose = np.array([pose['position']['x'], 
                     pose['position']['y'], 
                     pose['position']['z'],
                     pose['rotation']['x'], 
                     pose['rotation']['y'], 
                     pose['rotation']['z'], 
                     pose['rotation']['w'] 
                     ])
    return pose
    
def pose_to_transformation(pose):

    '''
    Convert poses to transformation matrix
    '''


    temp_0 = pose[0]
    temp_1 = pose[1]
    temp_2 = pose[2]

    # quaternions
    temp_x = pose[3]
    temp_y = pose[4]
    temp_z = pose[5]
    temp_w = pose[6]

    pose[4:6] *= -1
    pose[0] *= -1

    # print(pose)
    rot_mat = quat2mat(pose[3:])
    translation_vector = np.array([[pose[0]], [pose[1]], [pose[2]]]) / 1000  
    print(translation_vector)
    # translation_offset = np.array([[2.25], [-1.25], [0.5]])
    # translation_offset = np.array([[0.0], [-0.5], [0.0]])
    
    rot_mat_2 = np.array([[ 0,  1, 0, 0],
                          [-1,  0, 0, 0],
                          [ 0,  0, 1, 0],
                          [ 0,  0, 0, 1]])
    
    flip_x = np.eye(4)
    flip_x[0, 0] *= -1

    trans = flip_x @ rot_mat_2
    translation_offset = np.ones((3,1)) * 1


    # different transformation matrix
    transformation_mat = np.vstack((np.hstack((rot_mat, translation_vector + 0.5 ) ), np.array([0, 0, 0, 1]))) 
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,  rot_mat.T @ translation_vector)), np.array([0, 0, 0, 1])))
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,   (translation_vector) )), np.array([0, 0, 0, 1])))
    
    # translation_offset = -np.array([[1.0], [1.0], [2.5]])
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,  rot_mat.T @ (translation_offset)  + translation_vector)), np.array([0, 0, 0, 1])))
    
    print(transformation_mat.shape)
    return transformation_mat @ trans



# def custom_draw_geometry(pcd):

#     vis = o3d.visualization.Visualizer()
    
#     vis.create_window()
#     pcd_origin = o3d.geometry.PointCloud()
#     pcd_origin.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0],
#                                                              [0, 0, 1],
#                                                              [1, 0, 0]]))
#     pcd_origin.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
#     vis.add_geometry(pcd)
#     vis.add_geometry(pcd_origin)
    
#     ctr = vis.get_view_control()
#     print(ctr)
#     vis.run()

def read_json_file(path):
    '''
    Read json file
    '''
    json_data = []

    with open(path) as fp:
        for json_object in fp:
            json_data.append(json.loads(json_object))
    
    return json_data


def convert_tensor_to_numpy(t):

    return t.detach().cpu().numpy()

def read_depth(path, size = None):

    d = cv2.imread(path,  -1)

    if size is not None:
        d = np.array(Image.fromarray(d).resize((size[0], size[1])))

    return d

def preprocess_depth(depth):

    depth = depth.max() - depth
    depth = depth / depth.max()
    depth[depth > 0.8] = 1000000.0

    return depth



class CoordinateDistances:

    def __init__(self):
        self.coord_distance_dict = {}
        self.theta = torch.eye(3)[:2][None]
    
    @torch.no_grad()
    def get_coord_distance(self, size, device="cuda"):
        if size not in self.coord_distance_dict:
            grid = torch.nn.functional.affine_grid(self.theta, (1, 1, size, size), align_corners=None)# 1, H, W, 2
            # print(grid.shape)
            d = grid.reshape(1, -1, 2)
            # print(d.shape)
            dist = torch.sqrt(torch.sum(torch.square(d[:, :, None] - d[:, None]), -1) + 1e-12).to(device) # 1, hw, hw

            self.coord_distance_dict[size] = dist

        return self.coord_distance_dict[size]

coord_distance_depth = CoordinateDistances()

def fill_background_depth(depth, mask_in, device = "cuda"):


    # Borrowed from - https://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays
    x,y=np.mgrid[0:depth.shape[0],0:depth.shape[1]]
    mask = (mask_in[0, 0] < 0.5)

    xygood = np.array((x[~mask],y[~mask])).T
    xybad = np.array((x[mask],y[mask])).T

    # print(depth)
    depth[mask] = depth[~mask][KDTree(xygood).query(xybad)[1]]
    # print(depth)



    

    return depth
    # exit()

def get_transform_coordinates(image, depth, obj_mask = None, transform_in = torch.eye(4), use_softsplat = True, focal_length = 550, return_mesh=False):

    K = camera_matrix(focal_length, focal_length, image.shape[1] / 2.0, image.shape[0] / 2.0)
    # print(depth.min(), depth.max())
    # depth = depth.max() - depth
    # Constant depth case
    if np.sum(depth) == 0.5 * (depth.shape[0] * depth.shape[1]):
        depth = np.ones_like(depth) * 0.5

    else:
        # Normalize depth
        depth = depth / (depth.max() + 1e-8)
        # making sure that the closest point is far enough from the camera to rotate without causing smearing
        # depth = depth
        depth[depth > 0.95] = 1.0
        # depth = 1.0 / depth
        # print(depth.min(), depth.max())
        # depth = depth * 10

    mask = (depth < 0.95) * 1.0

    if obj_mask is not None:
        mask = obj_mask * mask


    mean_depth = np.mean(depth[mask >= 0.5])
    mask_torch = (torch.tensor(mask)[None, None] >= 0.5) * 1.0

    # depth = fill_background_depth(depth, mask_torch)

    # print(depth[mask >= 0.5].min(), depth[mask >= 0.5].max(), " min max")
    # print(mean_depth, depth.shape, " mean")

    # degrees = args.degrees
    # axis = args.axis

    # pose_transform = rotateAxis(degrees, 1)[None] @ rotateAxis(degrees, 2)[None]

    # pose_transform = rotateAxis(degrees, axis)[None]

    # pose_transform_2 = rotateAxis(0, 0)[None]

    # translate_transform = translateMatrix(0.0, 0.0, -mean_depth)
    
    # translate_transform_2 = translateMatrix(0, 0.0, 0).type_as(translate_transform)
    
    # # .type_as(translate_transform)

    # # pose_transform = translate_transform.inverse() @ pose_transform @ translate_transform
    # pose_transform = translate_transform.inverse() @ transform_in.type_as(translate_transform) @ translate_transform
    # print(transform_in, pose_transform)
    # print("Pose transform: \n ", pose_transform)

    pose_transform = transform_in
    image = torch.from_numpy(image)[None].permute(0, 3, 1, 2)

    with torch.no_grad():
        if use_softsplat:

            if return_mesh:
                projected_image, valid_points, depth_projected, transform_coordinates, mask_projected_mesh = forward_splatting_pytorch3d_warp(image.to("cuda"), torch.from_numpy(depth)[None][None].to("cuda"), torch.from_numpy(K)[None].to("cuda"), pose_transform[None].to("cuda"), return_coordinates=True, obj_mask = mask_torch, return_mesh=True)
            else:
                projected_image, valid_points, depth_projected, transform_coordinates = forward_splatting_pytorch3d_warp(image.to("cuda"), torch.from_numpy(depth)[None][None].to("cuda"), torch.from_numpy(K)[None].to("cuda"), pose_transform[None].to("cuda"), return_coordinates=True, obj_mask = mask_torch, return_mesh=False)

        else:
            projected_image, valid_points, depth_projected, transform_coordinates = forward_warp(image, torch.from_numpy(depth)[None][None], torch.from_numpy(K)[None], pose_transform[None], return_coordinates=True)


    # # print(projected_image.shape, valid_points.shape)
    projected_image = projected_image * valid_points[:, None]
    np_image = np.clip(convert_tensor_to_numpy(projected_image[0].permute(1, 2, 0)), 0, 1)
    t_coords = convert_tensor_to_numpy(transform_coordinates[0])

    if return_mesh:
        return t_coords, np_image, convert_tensor_to_numpy(mask_projected_mesh)
    return t_coords, np_image




# rast_opt = torch.compile(rasterize_points)

pixel_coords = None


class RasterizePointsXYsBlending(nn.Module):
    """
    Borrowed from: https://github.com/facebookresearch/synsin/blob/501ec49b11030a41207e7b923b949fab8fd6e1b5/models/layers/z_buffer_layers.py#L12
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options

    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
        self,
        radius=1.3,
        # Change
        points_per_pixel=15,
        accumulation = 'alphacomposite',
        # Change
        tau = 1.0,
        rad_pow = 2,

    ):
        super().__init__()
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.accumulation = accumulation
        self.tau = tau
        self.rad_pow = rad_pow
        self.rasterization_dict = {}
    
    def clear_cache(self):
        self.rasterization_dict = {}
    
    @torch.autocast("cuda")
    @torch.no_grad()
    def forward(self, pts3D, src):
#         pts3D - b, N, 3
#         src - b, F, N

        # print(self.tau, self.points_per_pixel, self.radius)
        
        pts3D = pts3D.to(torch.float32)
        src = src.to(torch.float32)
        bs = src.size(0)
        image_size = np.sqrt(src.shape[-1])

        # Make sure these have been arranged in the same way
        assert pts3D.size(2) == 3
        assert pts3D.size(1) == src.size(2)  
        
#         Convert to pytorch 3d coordinate frame where x - left, y - up, z - front
        pts3D[:,:,1] = - pts3D[:,:,1]
        pts3D[:,:,0] = - pts3D[:,:,0]
        
#         self.radius = image_size * 2
        radius = float(self.radius) / float(image_size) * 2.0

        # print(pts3D.dtype, src.dtype)
        pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))

        im_size = int((image_size))

        # with torch.autocast("cuda"):

        # Using cached points
        # if im_size in self.rasterization_dict:
        #     points_idx = self.rasterization_dict[im_size]["points_idx"][:1].repeat(bs, 1, 1, 1)
        #     dist = self.rasterization_dict[im_size]["dist"][:1].repeat(bs, 1, 1, 1)

        # else:
        points_idx, _, dist = rasterize_points(
                pts3D,  im_size, radius, self.points_per_pixel
            )
            # print(points_idx.shape, dist.shape)
            # exit()
            # self.rasterization_dict[im_size] = {}
            # self.rasterization_dict[im_size]["points_idx"] = points_idx.detach().clone()
            # self.rasterization_dict[im_size]["dist"] = dist.detach().clone()
#         if os.environ["DEBUG"]:
#             print("Max dist: ", dist.max(), pow(radius, self.opts.rad_pow))

        dist = dist / pow(radius, self.rad_pow)

#         if os.environ["DEBUG"]:
#             print("Max dist: ", dist.max())

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.tau)
            .permute(0, 3, 1, 2)
        )

        # print(alphas.shape)
        # transmittance = torch.clamp(1 - torch.cumsum(alphas, 1), min=0, max=1)
        # # appending 1 to the first transmittance value
        # ones_transmittance = torch.ones_like(transmittance[:, :1])
        # transmittance = torch.cat([ones_transmittance, transmittance[:, :-1]], 1)
        # alphas_new = transmittance * alphas
        # alphas_new = alphas_new / (torch.sum(alphas_new, 1, keepdims=True) + 1e-8)
        # alphas = alphas_new
        # print(alphas_new.max(), alphas_new.min(), alphas_new.sum(1))
        # exit()

        # print(alphas.min(), alphas.max())

        if self.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.norm_weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
            
        

        return transformed_src_alphas.to(torch.half)


SPLATTER = RasterizePointsXYsBlending()


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

        rotation_mat = torch.tensor([[np.cos(radians), -np.sin(radians),           0,          0],
                                 [np.sin(radians),  np.cos(radians),           0,          0],
                                 [              0,                0,           1,          0],
                                 [              0,                0,           0,          1]])

    elif axis == 1: # y - axis

        rotation_mat = torch.tensor([[np.cos(radians),                0,  np.sin(radians),          0],
                                 [              0,                1,                0,          0],
                                 [-np.sin(radians),               0, np.cos(radians),          0],
                                 [              0,                0,                0,          1]])

    elif axis == 0: # x - axis


        rotation_mat = torch.tensor([[             1,                0,                0,          0],
                                [              0,  np.cos(radians), -np.sin(radians),          0],
                                [              0,  np.sin(radians),  np.cos(radians),          0], 
                                [              0,                0,                0,          1]])
    
    return rotation_mat

def translateMatrixFromVector(v):

    translation_matrix = torch.eye(4).type_as(v)
    # translation_matrix[0,3] += x
    # translation_matrix[1,3] += y
    # translation_matrix[2,3] += z
    translation_matrix[:3, 3] += v

    return translation_matrix#.type_as(torch.double)


def splatter_mesh(mesh, splatter, image_size):


    radius = float(1e-6) / float(2 * image_size)

    # print(pts3D.dtype, src.dtype)
    # pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))

    im_size = int((image_size))



    p_to_face, zbuf, bary_coords, dists = rasterize_meshes(mesh, im_size, radius, splatter.points_per_pixel, perspective_correct=True)
    

    # print(p_to_face.shape, zbuf.shape, bary_coords.shape, dists.shape)

    frags = Fragments(pix_to_face=p_to_face, zbuf = zbuf, bary_coords=bary_coords, dists=dists)
    # print(frags.pix_to_face.shape, frags.zbuf.shape, frags.bary_coords.shape, frags.dists)
    # print(frags.pix_to_face.shape, frags.zbuf.shape, frags.bary_coords.shape, frags.dists.shape)

    # points_idx, _, dist = rasterize_points(
    #         pts3D,  im_size, radius, self.points_per_pixel
    #     )

    # print(mesh.num_faces_per_mesh())
    transformed_src_alphas = mesh.sample_textures(frags) # 1, 1, 1, F


    # print(transformed_src_alphas.shape)

    transformed_src_alphas = transformed_src_alphas[:, :, :, 0].permute(0, -1, 1, 2)
    # exit()

    # dist = dist / pow(radius, splatter.rad_pow)

    # alphas = (
    #     (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
    #     .pow(splatter.tau)
    #     .permute(0, 3, 1, 2)
    # )

    # if splatter.accumulation == 'alphacomposite':
    #     transformed_src_alphas = compositing.alpha_composite(
    #         face_idx.permute(0, 3, 1, 2).long(),
    #         alphas,
    #         mesh.textures.sample_textures(face_idx).permute(1,0),
    #     )
    # elif splatter.accumulation == 'wsum':
    #     transformed_src_alphas = compositing.weighted_sum(
    #         face_idx.permute(0, 3, 1, 2).long(),
    #         alphas,
    #         pts3D.features_packed().permute(1,0),
    #     )
    # elif splatter.accumulation == 'wsumnorm':
    #     transformed_src_alphas = compositing.norm_weighted_sum(
    #         face_idx.permute(0, 3, 1, 2).long(),
    #         alphas,
    #         pts3D.features_packed().permute(1,0),
    #     )
        
    

    return transformed_src_alphas#.to(torch.half)





def get_coordinate_array(mask):
    # Get mapping from image pixels to points on mesh for faces

    # Filling with -1 for null in the beginning
    mapping = torch.ones_like(mask[0, 0]) * -1

    num_points = mapping[mask[0, 0] >= 0.5].shape[0]

    mapping[mask[0, 0] >= 0.5] = torch.linspace(0, num_points - 1, steps = num_points).type_as(mapping)#.to(torch.int64)
    
    # print(mapping.dtype, mapping[0, 0])
    mapping = mapping.long()

    return mapping

def get_indexing_grid(im_shape):


    h, w = im_shape

    x = torch.linspace(0, h-1, steps=h)
    y = torch.linspace(0, w-1, steps=w)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    return grid_x.long(), grid_y.long() # shape h,w


def create_triangles(im2pcd_mapping, grid_x, grid_y):

    # print(grid_x.max(), grid_x.min(), im2pcd_mapping.shape, grid_y.min(), grid_y.max())

    # tl - tr
    # |
    # bl
    tl = im2pcd_mapping[grid_x, grid_y] # h, w
    tr = im2pcd_mapping[grid_x, grid_y + 1] # h, w
    bl = im2pcd_mapping[grid_x + 1, grid_y] # h, w
    # print(tl.shape, tr.shape, bl.shape, "tl")
    # exit()
    
    face_idx = torch.stack([tl, tr, bl], 0) # 3, h, w

    #       tr
    #       |
    # bl - br
    br = im2pcd_mapping[grid_x + 1, grid_y + 1] # h, w
    face_idx_2 = torch.stack([bl, tr, br], 0) # 3, h, w

    face_idx = face_idx.reshape(3, -1)
    face_idx_2 = face_idx_2.reshape(3, -1)

    face_ids = torch.cat([face_idx, face_idx_2], -1)
    # print(face_ids.shape, face_ids.min(), face_ids.max(), "before mask")

    min_face_ids = torch.min(face_ids, 0).values

    # Remove faces outside the mask
    face_ids = face_ids[:, min_face_ids > -1]
    return face_ids.transpose(0, 1)

def get_mesh(src_camera_coords, obj_mask):
    '''
    scc - b, 3, h, w
    obj_mask - 1, 1, h, w
    '''
    # print("running mesh operation")

    im2pcd_mapping = get_coordinate_array(obj_mask) # h, w shape

    src_camera_coords_pytorch3d = src_camera_coords.clone()
    # Converting from x-right, y-down, z-front to x-left, y-up, z-front for pytorch3d
    src_camera_coords_pytorch3d[:, :2] = -src_camera_coords[:, :2]



    h, w = obj_mask.shape[-2:]
    grid_x, grid_y = get_indexing_grid((h-1, w-1))



    pts = src_camera_coords_pytorch3d[0, :, obj_mask[0, 0] >= 0.5].reshape(3, -1).transpose(0, 1).to(src_camera_coords_pytorch3d.device).to(torch.float32) # V, 3
    verts_rgb = torch.ones_like(pts[None]).to(src_camera_coords_pytorch3d.device)[..., :1] # 1, V, 3
    face_ids = create_triangles(im2pcd_mapping, grid_x, grid_y).to(src_camera_coords_pytorch3d.device) # F, 3
    
    textures = TexturesVertex(verts_features=verts_rgb)

    # Create a Meshes object
    mesh = Meshes(
        verts=[pts],   
        faces=[face_ids],
        textures=textures
    )

    # print(face_ids.shape, face_ids.min(), face_ids.max())
    
    return mesh



class GaussianSmoothing(nn.Module):
    """
    # Borrowed from: https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/utils/gaussian_smoothing.py
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels = 1, kernel_size = 3, sigma=None, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim

        if sigma is None:
            sigma = (kernel_size[0] // 2 * 2 / 6.0)    
        
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim


        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.kernel_size = kernel_size
        self.register_buffer('weight', kernel)
        self.groups = channels

        # print(self.weight.shape)
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input_tensor):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input_tensor, weight=self.weight.type_as(input_tensor), groups=self.groups, padding=self.kernel_size[0] // 2)





def forward_splatting_pytorch3d_warp(tgt_image, depth, intrinsics, pose_transform, return_coordinates = False, obj_mask = None, return_mesh = False):


    depth = depth.float()
    intrinsics = intrinsics.type_as(depth)
    if obj_mask is not None:
        obj_mask = obj_mask.type_as(depth)
    src_camera_coords = pixel2cam(depth, intrinsics.inverse())
    pose_transform = pose_vec2mat(pose_transform).type_as(depth)


    # print(src_camera_coords.shape, pose_transform.shape, " forward splatting warp") # b, 3, h, w x b, 4, 4


    # Find the center of the object to bring it to the origin for performing the transformation
    b, c, h, w = src_camera_coords.shape
    cam_coords_pcd = src_camera_coords.reshape(b, c, -1)


    mask_pcd = obj_mask.reshape(1, 1, -1)
    obj_center = torch.mean(cam_coords_pcd[:, :, mask_pcd[0, 0] >= 0.5], -1)

    # print(obj_center.shape, obj_center)
    # translateMatrix()
    translate_transform = translateMatrixFromVector(-obj_center[0])

    translate_transform = translate_transform.repeat(b, 1, 1)
    

    pose_transform = translate_transform.inverse() @ pose_transform.type_as(depth) @ translate_transform

    src_cam_to_tgt_cam = pose_transform.float()


    # exit()
    tgt_cam_2_proj = src_cam_to_tgt_cam[:, :3, :] # Bx3x3 Bx3x4
    rot, tr = tgt_cam_2_proj[:,:,:3], tgt_cam_2_proj[:,:,-1:]
    # print(src_camera_coords.dtype, rot.dtype, tr.dtype, intrinsics.dtype)
    tgt_pix_coords = cam2pixel_vanilla(src_camera_coords, rot, tr, intrinsics, norm_scale = True, return_z=True) # b, h, w, 3

    
        
#     print(tgt_pix_coords.shape)
    

    tgt_image = tgt_image.type_as(tgt_pix_coords[0])
    
    b, h, w, _2 = tgt_pix_coords.shape
    _, f, _, _ = tgt_image.shape
    
    if return_mesh:
        mesh = get_mesh(tgt_pix_coords.permute(0, -1, 1, 2), obj_mask)
        projected_mask = splatter_mesh(mesh, SPLATTER, image_size=h)
        # print(projected_mask.shape, "projected mask")
#     print(depth.max(), depth.min())
    
#     print(tgt_pix_coords.shape, tgt_image.shape)
    
    tgt_pix_coords_in = tgt_pix_coords.reshape(b, h*w, -1)
    tgt_image_in = tgt_image.reshape(b, f, h*w)
    depth_in = depth.reshape(b, -1, h*w)
    
    
#     print(tgt_pix_coords_in.shape, tgt_image_in.shape)
    projected_img = SPLATTER(tgt_pix_coords_in, tgt_image_in)
    depth_projected = SPLATTER(tgt_pix_coords_in, depth_in)
#     valid_points = tgt_pix_coords_in
    valid_points = tgt_pix_coords[..., :2].abs().max(dim=-1)[0] <= 1
    
#     print(projected_img.shape, "splatted point cloud")
#     projected_img = softsplat.softsplat(tenIn=tgt_image, tenFlow=tgt_pix_coords.permute(0, -1, 1, 2), tenMetric=torch.exp(-depth / (depth.max() + 1e-8)), strMode="soft")
#     depth_projected = softsplat.softsplat(tenIn=depth, tenFlow=tgt_pix_coords.permute(0, -1, 1, 2), tenMetric=torch.exp(-depth / (depth.max() + 1e-8)), strMode="soft")
#     valid_points = tgt_pix_coords.abs().max(dim=-1)[0] <= 1
#     return projected_img, None, None

    # tgt_pix_coords_out = 
    # print(tgt_pix_coords.shape)
    # exit()
    if return_coordinates:
        if return_mesh:
            return projected_img, valid_points, depth_projected, tgt_pix_coords, projected_mask
        return projected_img, valid_points, depth_projected, tgt_pix_coords
    
    if return_mesh:
        
        return projected_img, valid_points, depth_projected, projected_mask
    return projected_img, valid_points, depth_projected


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, intrinsics):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]

 



    # print(cam_coords_flat.shape, torch.mean(cam_coords_flat[0], -1))
    # input_coords = intrinsics @ cam_coords_flat # B, 3, H*W

    
    # X_in = input_coords[:, 0]
    # Y_in = input_coords[:, 1]
    # Z_in = input_coords[:, 2].clamp(min=1e-8)

    # X_in_norm = X_in / Z_in
    # Y_in_norm = Y_in / Z_in
    # Get mask for high depth coordinates
    # print(cam_coords_flat.shape, cam_coords_flat[:, -1])

    if proj_c2p_rot is not None:
        # print(proj_c2p_rot)
        pcoords = proj_c2p_rot.float() @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr.type_as(proj_c2p_rot)  # [B, 3, H*W]

    mask = (pcoords[:, -1] > 100.0)*1.0 # B, H * W
    # mask = (cam_coords_flat[:, -1] > 10.0)*1.0 # B, H * W
    

    # print(pcoords.shape, torch.mean(pcoords[0], -1))
    pcoords = intrinsics @ pcoords
    # print(pcoords.shape, torch.mean(pcoords[0], -1))

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-8)



    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]


    global pixel_coords
    B, C, H, W = pixel_coords.shape
    pixel_coords_input = pixel_coords.reshape(-1, 3, H * W)[:, :2] # 1, 2, H * W
    X_in_norm = (pixel_coords_input[:, 0] * 2.0) / (w - 1) - 1.0
    Y_in_norm = (pixel_coords_input[:, 1] * 2.0) / (h - 1) - 1.0

    # print(X_in_norm.max(), X_in_norm.min(), Y_in_norm.max(), Y_in_norm.min())
    

    pixel_coords_f = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    pixel_in_coords = torch.stack([X_in_norm, Y_in_norm], dim=2)  # [B, H*W, 2]
    mask = mask[..., None]
    pixel_coords_f = mask * pixel_in_coords + (1.0 - mask) * pixel_coords_f

    # # print()
    delta_X = pixel_coords_f - pixel_in_coords

    delta_X = delta_X.permute(0, 2, 1 ) # B, 2, H*W
    delta_X = delta_X.reshape(1, 2, h, w)
    
    
    pixel_grid_y = ((pixel_coords_f[..., 1] + 1.0) / 2 * (h - 1)).to(torch.int64).clamp(min=0, max = h-1)
    pixel_grid_x = ((pixel_coords_f[..., 0] + 1.0) / 2 * (w - 1)).to(torch.int64).clamp(min=0, max=w-1)
    b_id = (torch.ones_like(pixel_grid_y) * 0).to(torch.int64)

    
    p_i_y = ((pixel_in_coords[..., 1] + 1.0) / 2 * (h - 1)).to(torch.int64).clamp(min=0, max = h-1)
    p_i_x = ((pixel_in_coords[..., 0] + 1.0) / 2 * (w - 1)).to(torch.int64).clamp(min=0, max = w-1)
    

    idx_sorted = torch.argsort(Z[0], descending=True)
    delta_new = delta_X.index_put_((b_id[0], torch.zeros_like(b_id[0]), pixel_grid_y[0][idx_sorted], pixel_grid_x[0][idx_sorted]), delta_X[b_id[0][idx_sorted], 0, p_i_y[0][idx_sorted], p_i_x[0][idx_sorted]])

    delta_new = delta_new.index_put_((b_id[0][idx_sorted], torch.ones_like(b_id[0]), pixel_grid_y[0][idx_sorted], pixel_grid_x[0][idx_sorted]), delta_X[b_id[0][idx_sorted], 1, p_i_y[0][idx_sorted], p_i_x[0][idx_sorted]])
    
    pixel_coords_f = pixel_coords_f.reshape(b, h, w, 2)

    delta_new = delta_new.permute(0, 2, 3, 1).reshape(1, h*w, 2) # 1, h*w, 2

    pixel_coords_f = pixel_in_coords - delta_new
    # print(Z[0].shape, "Z")

    
    # print(pixel_coords_f.shape)

    # (pixel_coords_f[..., 0] - 1) * ()

    # pixel_grid_y = ((pixel_coords_f[..., 1] + 1.0) / 2 * (h - 1)).to(torch.int64).clamp(min=0, max = h-1)
    # pixel_grid_x = ((pixel_coords_f[..., 0] + 1.0) / 2 * (w - 1)).to(torch.int64).clamp(min=0, max=w-1)
    # b_id = (torch.ones_like(pixel_grid_y) * 0).to(torch.int64)


    return pixel_coords_f.reshape(b,h,w,2)



def cam2pixel_vanilla(cam_coords, proj_c2p_rot, proj_c2p_tr, intrinsics, norm_scale = True, return_z = False):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]



    if proj_c2p_rot is not None:
        # print(proj_c2p_rot)
        pcoords = proj_c2p_rot.float() @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr.type_as(proj_c2p_rot)  # [B, 3, H*W]

    mask = (pcoords[:, -1] > 100.0)*1.0 # B, H * W
    # mask = (cam_coords_flat[:, -1] > 10.0)*1.0 # B, H * W
    

    # print(pcoords.shape, torch.mean(pcoords[0], -1))
    pcoords = intrinsics @ pcoords
    # print(pcoords.shape, torch.mean(pcoords[0], -1))

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)



    if norm_scale:
        
        X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    else:
        print("no norm scale")
        X_norm = (X / Z)  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = (Y / Z)  # Idem [B, H*W]


    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]


    if return_z:
        pixel_coords = torch.stack([X_norm, Y_norm, Z], dim=2)
        return pixel_coords.reshape(b,h,w,3)
    # print(pixel_coords.reshape(b,h,w,2).shape)
    return pixel_coords.reshape(b,h,w,2)

    


def forward_splatting_warp(tgt_image, depth, intrinsics, pose_transform, return_coordinates = False):

    src_camera_coords = pixel2cam(depth, intrinsics.inverse())
    pose_transform = pose_vec2mat(pose_transform).type_as(depth)
    intrinsics = intrinsics.type_as(depth)

    src_cam_to_tgt_cam = pose_transform

    tgt_cam_2_proj = src_cam_to_tgt_cam[:, :3, :] # Bx3x3 Bx3x4
    rot, tr = tgt_cam_2_proj[:,:,:3], tgt_cam_2_proj[:,:,-1:]
    tgt_pix_coords = cam2pixel_vanilla(src_camera_coords, rot, tr, intrinsics, norm_scale = True) # b, h, w, 2

    tgt_image = tgt_image.type_as(tgt_pix_coords[0])
    print(tgt_image.max(), tgt_image.min())
    print(depth.min(), depth.max())


    projected_img = softsplat.softsplat(tenIn=tgt_image.to(torch.float32), tenFlow=tgt_pix_coords.permute(0, -1, 1, 2).to(torch.float32), tenMetric=None, strMode="avg")
    depth_projected = softsplat.softsplat(tenIn=depth.to(torch.float32), tenFlow=tgt_pix_coords.permute(0, -1, 1, 2).to(torch.float32), tenMetric=None, strMode="avg")


    projected_img = projected_img.type_as(tgt_image)
    depth_projected = depth_projected.type_as(tgt_image)
    # projected_img = softsplat.softsplat(tenIn=tgt_image, tenFlow=tgt_pix_coords.permute(0, -1, 1, 2), tenMetric=torch.exp(-depth), strMode="soft")
    # depth_projected = softsplat.softsplat(tenIn=depth, tenFlow=tgt_pix_coords.permute(0, -1, 1, 2), tenMetric=torch.exp(-depth), strMode="soft")
    valid_points = tgt_pix_coords.abs().max(dim=-1)[0] <= 1
    if return_coordinates:
        return projected_img, valid_points, depth_projected, tgt_pix_coords
    
    return projected_img, valid_points, depth_projected



def quat2mat(quat):

    x, y, z, w = quat[:,0], quat[:,1], quat[:,2], quat[:,3]

    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    n = w2 + x2 + y2 + z2
    x = x / n
    y = y / n
    z = z / n
    w = w / n
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([1 - 2*y2 - 2*z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, 1 - 2*x2 - 2*z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, 1 - 2*x2 - 2*y2], dim=1).reshape(B, 3, 3)
    return rotMat

def pose_vec2mat(vec):

    size_list = list(vec.size())

    if len(size_list) == 3:
        # if dimension is [B 4 4] for multiview blender dataset

        return vec
    else:
        # If dimension is [B 7] for multiview nocs dataset
        b = vec.size(0)
        translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
        rot = vec[:,3:]
        rot_mat = quat2mat(rot)  # [B, 3, 3]

        invert_mat = torch.eye(4)
        invert_mat[0, 0] *= -1
        invert_mat[1, 1] *= -1

        # Adding 0.5 offset for dataset
        transform_mat = torch.cat([rot_mat,   (translation) + 0.5], dim=2)  # [B, 3, 4]
        transform_mat = torch.cat([transform_mat, torch.tensor([[0,0,0,1]]).unsqueeze(0).expand(1,1,4).type_as(transform_mat).repeat(b, 1, 1)], dim=1) # [B, 4, 4]
        return transform_mat @ invert_mat.type_as(transform_mat)


def set_id_grid(depth):
    global pixel_coords
    b, _, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1).type_as(depth)  # [1, 3, H, W]
    pixel_coords.to(depth.device)

def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    b, _, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    pixel_coords = pixel_coords.to(depth.device)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv.type_as(depth) @ current_pixel_coords.type_as(depth))
    cam_coords = cam_coords.reshape(b, 3, h, w)
    return cam_coords * depth


def pose_2_matrix(transformation_mat):
    '''
    Function to convert pose to transformation matrix
    '''
    flip_x = torch.eye(4)
    flip_x[2, 2] *= -1
    flip_x[1, 1] *= -1

    # views = pose.size(0)

    

    flip_x = flip_x.type_as(transformation_mat)

    # 180 degree rotation around x axis due to blender's coordinate system
    return  transformation_mat @ flip_x


def forward_warp(tgt_image, depth, intrinsics, pose_transform, return_coordinates = False):

    src_camera_coords = pixel2cam(depth, intrinsics.inverse())
    pose_transform = pose_vec2mat(pose_transform).type_as(depth)
    intrinsics = intrinsics.type_as(depth)

    src_cam_to_tgt_cam = pose_transform
    # print(src_cam_to_tgt_cam)
    tgt_cam_2_proj = src_cam_to_tgt_cam[:, :3, :] # Bx3x3 Bx3x4
    rot, tr = tgt_cam_2_proj[:,:,:3], tgt_cam_2_proj[:,:,-1:]
    tgt_pix_coords = cam2pixel(src_camera_coords, rot, tr, intrinsics, )

    # print(tgt_pix_coords.shape, tgt_pix_coords, "pix coords")

    tgt_image = tgt_image.type_as(tgt_pix_coords[0])

    # b_i, h_i, w_i = tgt_pix_coords
    # ones_ = torch.ones_like(b_i)
    # projected_image = tgt_image.index_input((b_i[0], 0 * ones_[0], h_i[0], w_i[0]), tgt_image[b_i[0], 0 * ones_[0], h_i[0], w_i[0]])
    projected_img = F.grid_sample(tgt_image, tgt_pix_coords, padding_mode='zeros', align_corners=True, mode="bilinear")
    depth_projected = F.grid_sample(depth, tgt_pix_coords, padding_mode='zeros', align_corners=False, mode="bilinear")
    valid_points = tgt_pix_coords.abs().max(dim=-1)[0] <= 1

    if return_coordinates:
        return projected_img, valid_points, depth_projected, tgt_pix_coords


    return projected_img, valid_points, depth_projected


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


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--input_depth", type=str)
    parser.add_argument("-im", "--input_image", type=str)
    parser.add_argument("-o", "--output_directory", type=str, default="./outputs/debug_outputs/")
    parser.add_argument("-d", "--degrees", type=float, default=0.0)
    parser.add_argument("-a", "--axis", type=int, default=0)
    parser.add_argument("-m", "--mask", type=str, default=None)


    args = parser.parse_args()

    


    image = cv2.imread(args.input_image, -1) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0


        
    if args.input_depth.split(".")[-1] == "npz":
        depth = np.load(args.input_depth)["arr_0"]
        # [734.9394386120183, 734.9394386120183, 639.5, 359.5]
        K = camera_matrix(734.9394386120183, 734.9394386120183, 639.5, 359.5)
        if args.mask is not None:
            mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) / 255.0
            print(mask.min(), mask.max())
            image = image * mask[..., None]
            depth = depth * mask
        
        depth = depth / (depth.max() + 1e-8)
        depth[depth==0] = 1000000.0
        depth = depth.astype(np.float32)   
    else:
        depth = cv2.imread(args.input_depth,  -1)
        K = camera_matrix(550, 550, image.shape[1] / 2.0, image.shape[0] / 2.0)



        print(depth.min(), depth.max())
        depth = depth.max() - depth
        depth = depth / depth.max()
        depth[depth > 0.8] = 1000000.0
        # depth = 1.0 / depth
        print(depth.min(), depth.max())
    # depth = depth / depth.max() + 1e-8
    # show_image(depth / depth.max())
    # K = camera_matrix(550, 550, image.shape[1] / 2.0, image.shape[0] / 2.0)
    mask = (depth < 100.0) * 1.0
    
    output_directory_name = os.path.join(args.output_directory, "")
    # transform = torch.eye(4)


    # SAVE POINT CLOUD
    point_cloud = save_point_cloud(K,image, mask, depth, 1, output_directory_name, max_depth=np.inf)
    # o3d.visualization.draw_geometries([pcd])


    # print(np.mean(depth))


    # print(depth[0])
    # plt.imshow(depth)
    # plt.show()
    if args.input_depth.split(".")[-1] == "npz" and args.mask is not None:
        mean_depth = np.mean(depth[(depth > 0.001) * (depth < 1.0)])
    else:
        mean_depth = np.mean(depth[depth < 1.0])

    print(mean_depth)
    degrees = args.degrees
    axis = args.axis

    # pose_transform = rotateAxis(degrees, 1)[None] @ rotateAxis(degrees, 2)[None]

    pose_transform = rotateAxis(degrees, axis)[None]

    pose_transform_2 = rotateAxis(0, 0)[None]

    translate_transform = translateMatrix(0.0, 0.0, -mean_depth).type_as(pose_transform)
    
    translate_transform_2 = translateMatrix(0, 0.0, 0).type_as(pose_transform)
    
    # .type_as(translate_transform)

    pose_transform =  translate_transform_2 @ translate_transform.inverse() @ pose_transform_2 @ pose_transform @ translate_transform

    image = torch.from_numpy(image)[None].permute(0, 3, 1, 2)

    with torch.no_grad():
        projected_image, valid_points, depth_projected, transform_coordinates = forward_warp(image, torch.from_numpy(depth)[None][None], torch.from_numpy(K)[None], pose_transform, return_coordinates=True)


    # # print(projected_image.shape, valid_points.shape)
    projected_image = projected_image * valid_points[:, None]
    np_image = np.clip(convert_tensor_to_numpy(projected_image[0].permute(1, 2, 0)), 0, 1)

    t_coords = convert_tensor_to_numpy(transform_coordinates[0])
    # convert_tensor_to_numpy(projected_image[0].permute()
    np_depth = convert_tensor_to_numpy(depth_projected[0, 0])


    args.output_directory = os.path.join(args.output_directory,"")
    os.makedirs(args.output_directory, exist_ok=True)

    od = args.output_directory
    print(t_coords.shape)

    plt.imsave(od + "projected_image.png", np_image)
    np.save(od + "transform_coordinates.npy", t_coords)
    np.save(od + "depth.npy", np_depth)
    plt.imsave(od + "input.png", convert_tensor_to_numpy(image[0].permute(1, 2, 0)))
    np.save(od + "transform.npy", convert_tensor_to_numpy(pose_transform)[0])
    # plt.imsave()
    # print(np_image.shape)
    # plt.imshow(np_image)
    # plt.show()



