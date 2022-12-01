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
from configs.path_config import PathConfig, ScanNet_OBJ_CLASS_IDS
import vtk
from utils.scannet.visualization.vis_scannet import Vis_Scannet
import numpy as np
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils.pc_util import rotz
import pickle
import random
import trimesh
import seaborn as sns
from utils.shapenet.common import Mesh
from dvis import dvis

device = 'cuda:0'

class PoseGen(Dataset):
    def __init__(self, split="train",overfit=True):
        self.split = split
        self.overfit = overfit
        if self.split == 'train':
            self.data = pickle.load( open( "/home/kaly/research/RfDNet/datasets/scenegen_train.pkl","rb"))
        elif self.split=='val':
            self.data =pickle.load( open( "/home/kaly/research/RfDNet/datasets/scenegen_val.pkl","rb"))
        if self.overfit:
            self.data = self.data[0:32]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        a=-1
        b=1
        text_embeddings = torch.zeros((40,512))
        box3D = torch.zeros((40,7))
        clip_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=True)
        data = self.data[index]
        count=0
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
        verts_list = []
        faces_list = []
        gt_meshes = []
        for i in data['objects']:
            try:
                text_embed = clip_model.encode_text(clip.tokenize(i['object_data']['text']))
                pose = torch.tensor(i['object_data']['box3D'])
                text_embeddings[count] = text_embed.detach()
                box3D[count] = pose
                shapenet_model = os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path, i['object_data']['shapenet_catid'], i['object_data']['shapenet_id'] + '.off')
                # print("Loading")
                # mesh = IO.load_mesh(shapenet_model)
                tmesh = trimesh.load(shapenet_model)
                
                verts_rgb = torch.ones_like(torch.tensor(np.asarray(tmesh.vertices)),dtype= torch.float32)[None]  # (1, V, 3)
                textures = TexturesVertex(verts_features=verts_rgb)
                # verts = torch.tensor(np.asarray(mesh.verts_padded()),dtype=torch.float32)
                # faces = torch.tensor(np.asarray(mesh.faces_padded()),dtype=torch.float32)
                gt_meshes.append(Meshes(verts=[torch.tensor(np.asarray(tmesh.vertices),dtype=torch.float32).to(device)],faces=[torch.tensor(np.asarray(tmesh.faces),dtype=torch.float32).to(device)],textures=textures))
                # mesh = Meshes(verts=[torch.tensor(np.asarray(tmesh.vertices),dtype=torch.float32).to(device)],faces=[torch.tensor(np.asarray(tmesh.faces),dtype=torch.float32).to(device)],textures=textures)
                
                # verts_list.append(mesh.verts_packed())
                # faces_list.append(mesh.faces_packed())
                count+=1
            except:
                continue
        if count<40:
            text_embeddings[count:40] = text_embeddings[count-1]
            box3D[count:40] = box3D[count-1]
            gt_meshes.extend([gt_meshes[-1]]*(40-count))
            #verts_list.extend([verts_list[-1]]*(40-count))
            #faces_list.extend([faces_list[-1]]*(40-count))
        text_embeddings = text_embeddings.permute(1,0)
        box3D = box3D.permute(1,0)
        box3D = a + ((box3D - min1[0]) * (b - a) / (max1[0] - min1[0])) #min-max norm to [-1,1]
        
        new_data={}
        new_data['pose']=box3D
        new_data['text'] = text_embeddings
        new_data['meshes'] = gt_meshes
        #new_data['verts_list'] = verts_list
        #new_data['faces_list'] = faces_list
        return new_data
    
    def __len__(self):
        return len(self.data)