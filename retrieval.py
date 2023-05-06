from diffusion_model import Unet1D,GaussianDiffusion1D, Trainer
from posegen_v2 import PoseGen as pgen2
from posegen import PoseGen as pgen1
from visualize_test import denormalize,get_scene_list,render_top_view
import trimesh
import torch
from tqdm import tqdm
import numpy as np
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from matplotlib import pyplot as plt
from bbox_utils import get_object_info_from_bounding_box, get_3d_box, box3d_iou, separate_boxes,detect_collisions
from torchmetrics.functional import kl_divergence
from glob import glob
import pickle
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path
import csv
import pandas as pd

# fid = FrechetInceptionDistance(feature=64,normalize = True)
# kid = KernelInceptionDistance(subset_size=50, normalize=True)

def retrieve_bboxmeshes(prediction,database):

    rows = prediction[3:6].detach().cpu().numpy()[:, None, :]

    columns = database.detach().cpu().numpy()[None, :, :] #1x12383x3


    product_matrix = rows * columns

    dot_matrix = np.sum(product_matrix, axis = 2)

    row_norm = np.linalg.norm(rows, axis = 2)
    column_norm = np.linalg.norm(columns, axis = 2)
    norm_matrix = row_norm * column_norm

    similarity_matrix = np.arccos(dot_matrix / norm_matrix)

    neighbours = np.argsort((similarity_matrix), axis = 0)
    return similarity_matrix, neighbours

retrieval_set = pd.read_csv('/home/kaly/research/text2scene/datasets/retrieval_database.csv',usecols=['size','path','cls_id'])
device = 'cuda:0'

pred = torch.tensor(np.load('/home/kaly/research/text2scene/datasets/pred.npy')).to(device)
pred[:,:8,:] = denormalize(pred[:,:8,:])
pred[:,7:9,:] = torch.round(pred[:,7:9,:])
print(retrieval_set)


# rfd_dir = glob("/home/kaly/research/RfDNet/datasets/scannet/processed_data/*")
# size = ['size']
# path = ['path']
# cls_id = ['cls_id']
# for scene in tqdm(rfd_dir):
#     try:
#         data = pickle.load( open(scene+ "/bbox.pkl","rb"))
#     except:
#         continue
#     for object in data:
#       #list of dictionaries of objects
#         try:
#             size.append(object['box3D'][3:6])
#             path.append(os.path.join('/home/kaly/research/RfDNet/'+ShapeNetv2_Watertight_Scaled_Simplified_path, object['shapenet_catid'], object['shapenet_id'] + '.off'))
#             cls_id.append(object['cls_id']+1)
#         except:
#             continue
    
# retrieval_set = zip(size,path,cls_id)
# with open("/home/kaly/research/text2scene/datasets/retrieval_database.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(retrieval_set)






# # gt = torch.tensor(np.load('/home/kaly/research/text2scene/datasets/gt.npy')).to(device)    
# # gt[:,:8,:] = denormalize(gt[:,:8,:])

# # print(gt[6,:,1])
# print(pred[6,:,1])

