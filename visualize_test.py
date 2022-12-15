import torch
import numpy as np
import pytorch3d
import trimesh
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path
from utils.pc_util import rotz
import math 

import pytorch3d
# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
from matplotlib import pyplot as plt 

torch.pi = math.pi

from pytorch3d.structures import join_meshes_as_scene

import cv2
device = 'cuda'
import warnings
warnings.filterwarnings('ignore')
  
def denormalize(pose):
    a = 0
    b = 1
    max1 = torch.tensor([[[3.7593],
                 [7.5061],
                 [2.1705],
                 [3.3867],
                 [4.4751],
                 [2.8320],
                 [3.1415]]]).to(device)
    min1 = torch.tensor([[[-3.7312],
             [-7.5617],
             [-0.3567],
             [ 0.0000],
             [ 0.0000],
             [ 0.0000],
             [-3.1416]]]).to(device)
    # max1 = torch.tensor(7.5061).to(device)
    # min1 = torch.tensor(-7.5617).to(device)
    pose = min1[0]+ ((pose -a) * (max1[0]-min1[0])/(b-a))
    #pose = min1+ ((pose -a) * (max1-min1)/(b-a))
    return pose

def render_top_view(scene_mesh,show=False):
    cameras = FoVPerspectiveCameras(device=device)
    
    raster_settings = RasterizationSettings(
        image_size=224, 
        blur_radius=0.0, 
        faces_per_pixel=1,
        bin_size=0 
    )
   
    # We can add a point light in front of the object. 
    lights = PointLights(device=device, location=((0.0, 0.0, 3.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,

            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    distance = 12   # distance from camera to the object
    elevation = 0.0   # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    R, T = torch.tensor(R,dtype=torch.float32), torch.tensor(T,dtype=torch.float32)
    # Render the teapot providing the values of R and T. 
    #images=renderer.render_front_views(kal_mesh, num_views=8, std=8, center_elev=0, center_azim=0, show=True, lighting=True,background=None, mask=False, return_views=False)

    image_ref = phong_renderer(meshes_world=scene_mesh, R=R, T=T)

    if show:
        image_ref1 = image_ref.cpu().numpy()
        plt.figure(figsize=(20,20))
        plt.subplot(1, 2, 1)
        plt.imshow(image_ref1.squeeze())
        plt.grid(False)
    return image_ref

def get_scene_list(poses, scenes):
     
    
    poses = denormalize(poses)
    poses = poses.permute(0,2,1)
    FLAP_YZ = False
    FLAP_XZ = False
    AUGMENT_ROT = False
    

    if AUGMENT_ROT:
        rot_angle = (np.random.random() * np.pi / 2) - np.pi / 4
    else:
        rot_angle = 0.
    rot_mat = torch.tensor(rotz(rot_angle)).to(device)
    rot_angle = torch.tensor(rot_angle).to(device)
    
    

    transform_m = torch.tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    
    outer_count=0
    scene_list=[]
    for scene in scenes:
        obj_list=[]
        inner_count = 0
        for obj in scene:
            if inner_count>0 and torch.eq(poses[outer_count][inner_count],poses[outer_count][inner_count-1])[0]:
                break

            if FLAP_YZ:
                poses[outer_count][inner_count][0] = -1 * poses[outer_count][inner_count][0]
                poses[outer_count][inner_count][6] = torch.sign(poses[outer_count][inner_count][6]) * torch.pi - poses[outer_count][inner_count][6]
            if FLAP_XZ:
                poses[outer_count][inner_count][1] = -1 * poses[outer_count][inner_count][1]
                poses[outer_count][inner_count][6] = -1 * poses[outer_count][inner_count][6]

            poses[outer_count][inner_count][0:3] = torch.tensor(np.dot(poses[outer_count][inner_count][0:3].detach().cpu().numpy(), np.transpose(rot_mat.cpu().numpy())))
            poses[outer_count][inner_count][6] += rot_angle

            '''Normalize angles to [-np.pi, np.pi]'''
            poses[outer_count][inner_count][6] = torch.tensor(torch.remainder(poses[outer_count][inner_count][6] + torch.pi, 2 * torch.pi) - torch.pi)

            tmp = obj.clone().detach()


            obj_points = tmp.verts_packed().detach().cpu().numpy()


            #front_view_image = render_front_view(mesh_torch,show=True)


            '''Fit obj points to bbox'''
            obj_points = obj_points - (obj_points.max(0) + obj_points.min(0))/2.
            obj_points = obj_points.dot(transform_m.T)
            obj_points = torch.tensor(obj_points.dot(np.diag(1/(obj_points.max(0) - obj_points.min(0)))).dot(np.diag(poses[outer_count][inner_count][3:6].detach().cpu().numpy())))
            orientation = poses[outer_count][inner_count][6]
            axis_rectified = torch.tensor([[torch.cos(orientation), torch.sin(orientation), 0], [-torch.sin(orientation), torch.cos(orientation), 0], [0, 0, 1]])
            obj_points = np.array(obj_points.detach().cpu().numpy().dot(axis_rectified.detach().cpu().numpy())) + poses[outer_count][inner_count][0:3].detach().cpu().numpy()

            verts_rgb = torch.ones_like(torch.tensor(obj_points),dtype= torch.float32)[None]  # (1, V, 3)
            textures = TexturesVertex(verts_features=verts_rgb.to(device))

            obj_list.append(Meshes(verts=[torch.tensor(obj_points,dtype=torch.float32).to(device)],faces=[torch.tensor(tmp.faces_packed(),dtype=torch.float32).to(device)],textures=textures))

            

            inner_count+=1
        scene_list.append(join_meshes_as_scene(obj_list))
        outer_count+=1
    return scene_list