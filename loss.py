from torch.utils.data import Dataset,DataLoader
import clip 
import torch
import pickle
import pickle
import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from pytorch3d.io import IO

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes
from torchvision import transforms
# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.datasets import collate_batched_meshes
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
import numpy as np
import sys
sys.path.append('.')
import os
from torch import _VF


import numpy as np
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path

from utils.pc_util import rotz
import pickle
import random
import trimesh
import seaborn as sns
from utils.shapenet.common import Mesh
from visualize_test import render_top_view, get_scene_list

device = 'cuda:0'
from torchvision import transforms

def render_loss(pred,gt,meshes):
    pred_scene_list = get_scene_list(pred,meshes) 
    gt_scene_list = get_scene_list(gt,meshes)

    pred_images = torch.zeros((pred.shape[0],224,224,4))
    gt_images = torch.zeros((pred.shape[0],224,224,4))

    count = 0
    for j in pred_scene_list:
        pred_images[count] = torch.tensor(render_top_view(j))
        count+=1
    
    count = 0
    for j in gt_scene_list:
        gt_images[count] = torch.tensor(render_top_view(j))
        count+=1

    criterion = nn.MSELoss()
    return criterion(pred_images[:,:,:,:3],gt_images[:,:,:,:3])

def clip_loss(text,pred,meshes,clip_model,preprocess):
    res =224
    pred_scene_list = get_scene_list(pred,meshes) 
    pred_images = torch.zeros((pred.shape[0],224,224,4))
    count = 0
    for j in pred_scene_list:
        pred_images[count] = render_top_view(j)
        count+=1
    img_features = []
    for i in pred_images:
        pred_feats = preprocess(transforms.ToPILImage()(i[:,:,:3])).to(device)
        img_features.append(clip_model.encode_image(pred_feats.unsqueeze(0)).repeat(1,40,1))
    
    img_features = torch.stack(img_features).to(device).squeeze(1)
    img_features = img_features.permute(0,2,1)
    #print(img_features.shape)
    criterion = nn.CosineSimilarity()
    loss = torch.mean(criterion(img_features,text))
    return loss

