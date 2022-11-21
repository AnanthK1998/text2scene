import torch
import numpy as np
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
import trimesh
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path
from utils.pc_util import rotz
from matplotlib import pyplot as plt
from pytorch3d.structures import join_meshes_as_scene

import cv2

device = "cuda:0"

def denormalize(pose):
    a = -1
    b = 1
    max1 = torch.tensor([[[3.7593],
                 [7.5061],
                 [2.1705],
                 [3.3867],
                 [4.4751],
                 [2.8320],
                 [3.1415]]])
    min1 = torch.tensor([[[-3.7312],
             [-7.5617],
             [-0.3567],
             [ 0.0000],
             [ 0.0000],
             [ 0.0000],
             [-3.1416]]])
    pose = min1[0]+ ((pose -a) * (max1[0]-min1[0])/(b-a))
    return pose



def get_scene_list(poses, objects):
     
    poses = denormalize(poses)
    poses = poses.permute(1,0)
    FLAP_YZ = False
    FLAP_XZ = False
    AUGMENT_ROT = False
    

    if AUGMENT_ROT:
        rot_angle = (np.random.random() * np.pi / 2) - np.pi / 4
    else:
        rot_angle = 0.
    rot_mat = rotz(rot_angle)
    rot_angle = torch.tensor(rot_angle)
    
    

    transform_m = torch.tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    scene=[]
    count=0
    scene_list=[]
    for mesh in objects:

        if FLAP_YZ:
            poses[count][0] = -1 * poses[count][0]
            poses[count][6] = torch.sign(poses[count][6]) * torch.pi - poses[count][6]
        if FLAP_XZ:
            poses[count][1] = -1 * poses[count][1]
            poses[count][6] = -1 * poses[count][6]

        poses[count][0:3] = torch.tensor(np.dot(poses[count][0:3].numpy(), np.transpose(rot_mat)))
        poses[count][6] += rot_angle

        '''Normalize angles to [-np.pi, np.pi]'''
        poses[count][6] = torch.tensor(np.mod(poses[count][6].numpy() + np.pi, 2 * np.pi) - np.pi)

        
        
        temp = mesh
        obj_points = mesh.verts_packed().cpu().numpy()
        mesh_points = obj_points
        
        #front_view_image = render_front_view(mesh_torch,show=True)


        '''Fit obj points to bbox'''
        obj_points = obj_points - (obj_points.max(0) + obj_points.min(0))/2.
        obj_points = obj_points.dot(transform_m.T)
        obj_points = obj_points.dot(np.diag(1/(obj_points.max(0) - obj_points.min(0)))).dot(np.diag(poses[count][3:6].numpy()))
        orientation = poses[count][6].numpy()
        axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
        obj_points = obj_points.dot(axis_rectified) + poses[count][0:3].numpy()
        
        verts_rgb = torch.ones_like(torch.tensor(obj_points),dtype= torch.float32)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        temp= Meshes(verts=[torch.tensor(obj_points,dtype=torch.float32).to(device)],faces=[mesh.faces_packed().to(device)],textures=textures)
        scene_list.append(temp)

        

        count+=1
    return scene_list

def render_top_view(scene,show=False):
    cameras = FoVPerspectiveCameras(device=device)
    scene_mesh = join_meshes_as_scene(scene)
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
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
    distance = 8   # distance from camera to the object
    elevation = 0.0   # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

   

    image_ref = phong_renderer(meshes_world=scene_mesh, R=R, T=T)

    if show:
        image_ref1 = image_ref.cpu().numpy()
        plt.figure(figsize=(20,20))
        plt.subplot(1, 2, 1)
        plt.imshow(image_ref1.squeeze())
        plt.grid(False)
    return image_ref