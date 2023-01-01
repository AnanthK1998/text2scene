from diffusion_model import Unet1D,GaussianDiffusion1D, Trainer
from posegen_v2 import PoseGen as pgen2
from posegen import PoseGen as pgen1
from visualize_test import denormalize,get_scene_list
import trimesh
import torch
from tqdm import tqdm
import os

device = 'cuda:0'
model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 7,
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

mode = 'val'


milestone = 221#135#128-best#125#115#105 
path = '/home/kaly/research/text2scene/results/'+str(milestone)
if not os.path.exists(path):
    os.mkdir(path)
trainer.load(milestone) #21
data = pgen2(mode)
dl = torch.utils.data.DataLoader(data,batch_size=40,shuffle=False)
for i, (cond,gt) in tqdm(enumerate(dl)):
    #cond = data[idx][0].unsqueeze(0).to(device)
    cond = cond.to(device)
    gt = gt.to(device)
    sampled_seq = diffusion.sample(cond,batch_size = 40)
    
    break
#[:,:,:size-1]
counter =0
for idx in tqdm(range(0,40)):
    size = pgen1(mode)[idx]['size']
    #sampled_seq = denormalize(sampled_seq)
    # gt = data[idx][1].unsqueeze(0).to(device)
    meshes = [pgen1(mode)[idx]['meshes'][:size-1]]
    pred = sampled_seq[idx].unsqueeze(0).to(device)[:,:,:size-1]
    
    ground = gt[idx].unsqueeze(0).to(device)[:,:,:size-1]
    
    # print(sampled_seq.shape)
    # print(gt.shape)
    pred_scene_list = get_scene_list(pred,meshes)
    gt_scene_list = get_scene_list(ground,meshes)

    count = 0 
    for j in pred_scene_list:
            
        #images[count] = torch.tensor(render_top_view(j))
        verts= j.verts_packed().cpu().numpy()
        faces = j.faces_packed().cpu().numpy()
        temp = trimesh.Trimesh(vertices =verts,faces=faces)
        temp.export(path+ '/pred_{}_{}-{}_new.obj'.format(mode,idx,milestone))
        count+=1

        
        
    count=0
    for j in gt_scene_list:
        
        verts= j.verts_packed().cpu().numpy()
        faces = j.faces_packed().cpu().numpy()
        temp = trimesh.Trimesh(vertices =verts,faces=faces)
        temp.export('/home/kaly/research/text2scene/results/gt/gt_{}_{}.obj'.format(mode,idx))
        #images[count] = torch.tensor(render_top_view(j))
        count+=1