from diffusion_model import Unet1D,GaussianDiffusion1D, Trainer
from posegen_v2 import PoseGen as pgen2
from posegen import PoseGen as pgen1
from visualize_test import denormalize,get_scene_list
import trimesh
import torch
from tqdm import tqdm
import numpy as np
import os

import clip 

clip_model,_ = clip.load('ViT-B/32', 'cpu', jit=True)
device = 'cuda:0'
model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 8,
    self_condition=True
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
    amp = False                       # turn on mixed precision
)

mode = 'train'

s = 22

text= ['the chair is at the right end of the table. it is to the right of another chair. it is the last chair on this side of the table.',
'this is a brown chair. it is turned toward the front of the table.',
'there is a chair pushed up to the table. it is the second from the left.',
'a brown chair. it is placed next to a table. behind it is the third chair.',
'a brown chair, it is placed next to a table. in the left it is the second chair.',
'this is a brown chair. it is at the table.',
'this is a long table. it is surrounded by chairs.',
'this is a tall chair. it is at the head of the table.',
'there are brwon wooden cabinets. placed behind the kitchen.',
'this is a black kitchen counter. it is on top of a kitchen cabinet.',
'this is a black tv. it is mounted to the wall.',
'this is the long skinny table under the painting on the wall .  the painting upon the wall is gray .  the wall the painting is on its opposite the doors .',
'this is a gray trash can. it is to the left of a table.']

# cond = torch.zeros((1,512,40))
# for i in text: 
permute = True
milestone = 45#221#135#128-best#125#115#105 
path = '/home/kaly/research/text2scene/results/'+str(milestone)
if not os.path.exists(path):
    os.mkdir(path)
trainer.load(milestone) #21
data = pgen2(mode)

dl = torch.utils.data.DataLoader(data,batch_size=1,shuffle=False)
for i, (cond,gt) in tqdm(enumerate(dl)):
    #cond = data[idx][0].unsqueeze(0).to(device)
    cond = cond.to(device)
    p = np.random.permutation(s) #22
    p_list = p.tolist()
    if permute:
        cond[:,:,:s]= cond[:,:,p]
    # temp = cond[:,:,0]
    # cond[:,:,0]= cond[:,:,-1]
    # cond[:,:,-1]= temp
    #cond = torch.zeros((1,512,40)).to(device)
    gt = gt.to(device)
    sampled_seq = diffusion.sample(cond,batch_size = 1)
    
    break
print(sampled_seq.shape)
#[:,:,:size-1]
counter =0
for idx in tqdm(range(0,1)):
    size = pgen1(mode)[idx]['size']
    #sampled_seq = denormalize(sampled_seq)
    #gt = data[idx][1].unsqueeze(0).to(device)
    objs = pgen1(mode)[idx]['meshes']
    if permute:
        objs = [objs[i] for i in p_list]
    meshes = [objs]
    
    pred = sampled_seq[idx].unsqueeze(0).to(device)[:,:,:size]
    #ground = gt[idx].unsqueeze(0).to(device)[:,:,:size]
    # temp = ground[:,:,0]
    # ground [:,:,0] = ground[:,:,-1]
    # ground[:,:,-1] = temp
    # print(sampled_seq.shape)
    # print(gt.shape)
    pred_scene_list = get_scene_list(pred[:,:7,:],meshes)
    #gt_scene_list = get_scene_list(ground[:,:7,:],meshes)

    count = 0 
    for j in pred_scene_list:
            
        #images[count] = torch.tensor(render_top_view(j))
        verts= j.verts_packed().cpu().numpy()
        faces = j.faces_packed().cpu().numpy()
        temp = trimesh.Trimesh(vertices =verts,faces=faces)
        if permute:
            temp.export(path+ '/pred_{}_{}-{}_permuted.obj'.format(mode,idx,milestone))
        else:
            temp.export(path+ '/pred_{}_{}-{}.obj'.format(mode,idx,milestone))
        count+=1

        
        
    # count=0
    # for j in gt_scene_list:
        
    #     verts= j.verts_packed().cpu().numpy()
    #     faces = j.faces_packed().cpu().numpy()
    #     temp = trimesh.Trimesh(vertices =verts,faces=faces)
    #     temp.export('/home/kaly/research/text2scene/testing/gt/gt_{}_{}.obj'.format(mode,idx))
    #     #images[count] = torch.tensor(render_top_view(j))
    #     count+=1