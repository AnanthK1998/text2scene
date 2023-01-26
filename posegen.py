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
import torchvision.transforms as T
import numpy as np
import sys
sys.path.append('.')
import os



import numpy as np
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path

from utils.pc_util import rotz
import pickle
import random
import trimesh
import seaborn as sns
from utils.shapenet.common import Mesh
from dvis import dvis

device = 'cuda:0'

import warnings
warnings.filterwarnings('ignore')

class PoseGen(Dataset):
    def __init__(self, split="train",overfit=False,use_cache= False):
        self.split = split
        self.overfit = overfit
        if self.split == 'train':
            self.data = pickle.load( open( "/home/kaly/research/RfDNet/datasets/scenegen_train.pkl","rb"))
        elif self.split=='val':
            self.data =pickle.load( open( "/home/kaly/research/RfDNet/datasets/scenegen_val.pkl","rb"))
        if self.overfit:
            self.data = self.data[0:32]
        self.clip_model,_ = clip.load('ViT-B/32', 'cpu', jit=True)
        self.use_cache = use_cache
        self.cached_data = []
        # self.transform = self.transform = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize((0.5),(0.5))
        # ])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        a=-1
        b=1
        text_embeddings = torch.zeros((40,512))
        box3D = torch.zeros((40,7))
        data = self.data[index]
        count=0
        max1 = torch.tensor([[[4.6341],
                 [7.5061],
                 [2.1705],
                 [3.3867],
                 [5.2126],
                 [2.8320],
                 [3.1415]]])
        min1 = torch.tensor([[[-3.9863],
                 [-7.5617],
                 [-0.3567],
                 [ 0.0000],
                 [ 0.0000],
                 [ 0.0000],
                 [-3.1416]]])
        max2= torch.tensor(7.0983) 
        min2= torch.tensor(-3.4024)
        # max1 = torch.tensor(7.5061) 
        # min1 = torch.tensor(-7.5617)
        gt_meshes = []
        if not self.use_cache:
            for i in data['objects']:
                try:
                    text_embed = self.clip_model.encode_text(clip.tokenize(i['object_data']['text']))
                    pose = torch.tensor(i['object_data']['box3D'])
                    text_embeddings[count] = text_embed.detach()
                    box3D[count] = pose
                    shapenet_model = os.path.join('/home/kaly/research/RfDNet/'+ShapeNetv2_Watertight_Scaled_Simplified_path, i['object_data']['shapenet_catid'], i['object_data']['shapenet_id'] + '.off')
                    
                    tmesh = trimesh.load(shapenet_model)

                    verts_rgb = torch.ones_like(torch.tensor(np.asarray(tmesh.vertices)),dtype= torch.float32)[None]  # (1, V, 3)
                    textures = TexturesVertex(verts_features=verts_rgb)
                    gt_meshes.append(Meshes(verts=[torch.tensor(np.asarray(tmesh.vertices),dtype=torch.float32)],faces=[torch.tensor(np.asarray(tmesh.faces),dtype=torch.float32)],textures=textures))
                    count+=1
                except (KeyError,RuntimeError) as e : #sometimes text token context length>77, hence runtime error. skip those
                    continue
            size = count+1
                
            text_embeddings = text_embeddings.permute(1,0)
            box3D = box3D.permute(1,0)
            box3D = a + ((box3D - min1[0]) * (b - a) / (max1[0] - min1[0])) #min-max norm to [-1,1]
            text_embeddings = a + ((text_embeddings - min2) * (b - a) / (max2 - min2))

            new_data={}
            new_data['pose']=box3D.squeeze(0)
            new_data['text'] = text_embeddings.squeeze(0)
            new_data['meshes'] = gt_meshes
            new_data['size'] = size
            self.cached_data.append(new_data)
        # else:
        #     new_data = self.cached_data[index]

        return new_data
    
    # def set_use_cache(self, use_cache):
    #     if use_cache:
    #         self.cached_data = torch.stack(self.cached_data)
    #     else:
    #         self.cached_data = []
    #     self.use_cache = use_cache


    
    