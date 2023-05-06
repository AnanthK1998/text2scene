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
fid = FrechetInceptionDistance(feature=64,normalize = True)
kid = KernelInceptionDistance(subset_size=50, normalize=True)

import clip


device = 'cuda:0'
clip_model,_ = clip.load('ViT-B/32', 'cpu', jit=True)

model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 9,
    self_condition=True,
    use_dgcnn=False,
    use_stn=False
).cuda()

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 40,
    timesteps = 1000,
    objective = 'pred_v'
).cuda()

trainer = Trainer(
    diffusion,
    train_batch_size = 16,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    results_folder = './pose_weights_baseline'
)

mode = 'val'

torch.manual_seed(42)



batchsize=65#65#130

milestone = 91#58-best#56#28
#221#135#128-best#125#115#105 
path = '/home/kaly/research/text2scene/results/'+str(milestone)
if not os.path.exists(path):
    os.mkdir(path)
trainer.load(milestone) #21
data = pgen2(mode,inference=True)

dl = torch.utils.data.DataLoader(data,batch_size=batchsize,shuffle=False)
for i, (cond,gt,size) in tqdm(enumerate(dl)):
    cond = cond.to(device)
    gt = gt.to(device)
    sampled_seq = diffusion.sample(cond,batch_size = batchsize)
    
    break
#print(sampled_seq.shape)
#[:,:,:size-1]
counter =0
pred_images = torch.zeros((batchsize,224,224,3),dtype=torch.uint8)
gt_images = torch.zeros((batchsize,224,224,3),dtype=torch.uint8)

# for idx in tqdm(range(batchsize)):
#     data = pgen1(mode)[idx]
#     size =data['size']
#     objs = data['meshes']
#     meshes = [objs]
#     bboxes = np.zeros((size,8,3))
#     pred = sampled_seq[idx].unsqueeze(0).to(device)[:,:,:size]
#     # for i in range(size):
#     #     bboxes[i] = get_3d_box(pred[0,3:6,i].cpu().numpy().tolist(),pred[0,6,i].cpu().numpy().tolist(),pred[0,0:3,i].cpu().numpy().tolist())
#     # collisions = detect_collisions(bboxes)
#     # for pair in collisions:
#     #     bboxes[pair[0]],bboxes[pair[1]]= separate_boxes(bboxes[pair[0]],bboxes[pair[1]],iou_threshold=0.17)
    
#     # for i in range(size):
#     #     center,scale = get_object_info_from_bounding_box(bboxes[i])
#     #     center,scale = torch.tensor(center).to('cuda:0'),torch.tensor(scale).to('cuda:0')
#     #     #pred[0,3:6,i] = scale
#     #     pred[0,0:3,i] = center
    
    


    
    

#     ground = gt[idx].unsqueeze(0).to(device)[:,:,:size]
    
#     pred_scene_list = get_scene_list(pred[:,:8,:],meshes)
#     gt_scene_list = get_scene_list(ground[:,:8,:],meshes)
    
#     count = 0 
#     for j in pred_scene_list:
        
#         pred_images[count] = torch.tensor(render_top_view(j.to('cuda:0'))[:,:,:,:3]*255,dtype=torch.uint8)
#         plt.imsave('/home/kaly/research/text2scene/test-render-pred.jpg',pred_images[count].cpu().numpy())
#         #print(torch.amax(pred_images[count]),torch.amin(pred_images[count]))
#         verts= j.verts_packed().cpu().numpy()
#         faces = j.faces_packed().cpu().numpy()
#         temp = trimesh.Trimesh(vertices =verts,faces=faces)
#         temp.export(path+ '/pred_{}_{}-{}.obj'.format(mode,idx,milestone))
#         count+=1

        
        
#     count=0
#     for j in gt_scene_list:
        
#         verts= j.verts_packed().cpu().numpy()
#         faces = j.faces_packed().cpu().numpy()
#         temp = trimesh.Trimesh(vertices =verts,faces=faces)
#         temp.export('/home/kaly/research/text2scene/testing/gt/gt_{}_{}.obj'.format(mode,idx))
#         gt_images[count] = torch.tensor(render_top_view(j)[:,:,:,:3]*255,dtype=torch.uint8)
#         plt.imsave('/home/kaly/research/text2scene/test-render-gt.jpg',gt_images[count].cpu().numpy())
#         count+=1


# pred_images = pred_images.permute(0,3,1,2)
# gt_images = gt_images.permute(0,3,1,2)

# clip_image_feats = clip_model.encode_image(pred_images).unsqueeze(1).repeat(1,40,1).to('cuda:0')
# data = pgen2(mode,inference=True,normalize_text=False)
# dl = torch.utils.data.DataLoader(data,batch_size=batchsize,shuffle=False)
# for i, (cond,_,size) in enumerate(dl):
#     cond = cond.permute(0,2,1).cpu().detach().numpy()
#     size = size.cpu().detach().numpy()
#     break

# text_embeddings = cond / np.linalg.norm(cond, axis=2, keepdims=True)
# clip_image_feats = clip_image_feats.cpu().detach().numpy() / np.linalg.norm(clip_image_feats.cpu().detach().numpy(), axis=1, keepdims=True)
# clip_score = 0.0
# for i in range(batchsize):
#     similarity_matrix = np.dot(text_embeddings[i][:size[i],:], clip_image_feats[i][:size[i],:].T)
#     clip_score += np.trace(similarity_matrix) / similarity_matrix.shape[0]

# clip_score = clip_score / batchsize

# fid.update(gt_images, real=True)
# fid.update(pred_images, real=False)
# fid_score = fid.compute()

# kid.update(gt_images, real=True)
# kid.update(pred_images, real=False)
# kid_score = kid.compute()

# #kl_score = kl_divergence()

# print('FID Score: ',fid_score)
# print('KID Score: ',kid_score)

# print('CLIP-Score: ', clip_score)