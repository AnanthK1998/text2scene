from diffusion_model import Unet1D,GaussianDiffusion1D, Trainer
from posegen_v2 import PoseGen as pgen2
from posegen import PoseGen as pgen1
from visualize_test import denormalize,get_scene_list
import trimesh

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

trainer.load(21)
data = pgen2('train')
cond = data[2][0].unsqueeze(0).to(device)
sampled_seq = diffusion.sample(cond,batch_size = 1)#[:,:,:12]
#sampled_seq = denormalize(sampled_seq)
gt = data[2][1].unsqueeze(0).to(device)#[:,:,:12]
meshes = [pgen1('train')[2]['meshes']]#[:12]]
# print(sampled_seq.shape)
# print(gt.shape)
pred_scene_list = get_scene_list(sampled_seq,meshes)
gt_scene_list = get_scene_list(gt,meshes)

count = 0 
for j in pred_scene_list:
        
        #images[count] = torch.tensor(render_top_view(j))
        verts= j.verts_packed().cpu().numpy()
        faces = j.faces_packed().cpu().numpy()
        temp = trimesh.Trimesh(vertices =verts,faces=faces)
        temp.export('/home/kaly/research/text2scene/results/pred_2_new.obj')
        count+=1

    
    
count=0
for j in gt_scene_list:
    
    verts= j.verts_packed().cpu().numpy()
    faces = j.faces_packed().cpu().numpy()
    temp = trimesh.Trimesh(vertices =verts,faces=faces)
    temp.export('/home/kaly/research/text2scene/results/gt_2.obj')
    #images[count] = torch.tensor(render_top_view(j))
    count+=1