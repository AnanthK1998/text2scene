from contextlib import contextmanager
from copy import deepcopy
import math
import clip
from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from tqdm import tqdm, trange
from statistics import mean
import glob
from PIL import Image
import os
from posegen import PoseGen
import wandb
from pytorch3d.datasets import collate_batched_meshes
#from loss import render_loss, clip_loss
from visualize_test import render_top_view, get_scene_list
device='cuda'
import trimesh

@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True,last_layer = False):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv1d(c_in, c_mid, 3, padding=1),
            nn.Dropout(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_mid, c_out, 3, padding=1),
            nn.Dropout(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.Tanh() if last_layer else nn.ReLU(inplace=True)
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[...,None].repeat(1,1,shape[2])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 128  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        #self.class_embed = nn.Embedding(4097, 4)

        self.net = nn.Sequential(   # 32x32
            ResConvBlock(7 + 512 + 16, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool1d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool1d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool1d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        nn.Upsample(scale_factor=2),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    nn.Upsample(scale_factor=2),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 7, dropout_last=False,last_layer=True),
        )

    def forward(self, input, log_snrs, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]),input.shape)
        #class_embed = expand_to_planes(self.class_embed(cond), input.shape)
        #print(input.shape, cond.shape, timestep_embed.shape)
        return self.net(torch.cat([input, cond, timestep_embed], dim=1))

# Define the noise schedule and sampling loop

def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given the log SNR for a timestep."""
    return log_snrs.sigmoid().sqrt(), log_snrs.neg().sigmoid().sqrt()


def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -torch.expm1(1e-4 + 10 * t**2).log()


@torch.no_grad()
def sample(model, x, steps, eta, classes):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * log_snrs[i], classes).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred

print('Using device:', device)
torch.manual_seed(0)
seed=0
train_set = PoseGen(overfit=True)

model = Diffusion().to(device)
model_ema = deepcopy(model)


rng = torch.quasirandom.SobolEngine(1, scramble=True)

ema_decay = 0.998

# The number of timesteps to use when sampling
steps = 4097

# The amount of noise to add each timestep when sampling
# 0 = no noise (DDIM)
# 1 = full noise (DDPM)
eta = 1.

@torch.no_grad()
@torch.random.fork_rng()
@eval_mode(model_ema)
def demo():
    tqdm.write('\nSampling...')
    torch.manual_seed(seed)
    net1 = torch.load('/home/kaly/research/text2scene/pose_weights/pose_diffusion_epoch_33.pth')['model']
    net2 = torch.load('/home/kaly/research/text2scene/pose_weights/pose_diffusion_epoch_33.pth')['model_ema']
    model.load_state_dict(net1)
    model_ema.load_state_dict(net2)
    noise = torch.randn([1,7,40], device=device)
    #fakes_classes = torch.arange(2, device=device).repeat_interleave(2, 0)
    fake_classes = torch.zeros((1,512,40),device=device)
    
    fake_classes[0] = train_set[0]['text'].unsqueeze(0)
    meshes = [train_set[0]['meshes']]
    fakes = sample(model_ema, noise, steps, eta, fake_classes).to(device)
    pred_scene_list = get_scene_list(fakes,meshes)
    gt = train_set[0]['pose'].unsqueeze(0).to(device)
    gt_scene_list = get_scene_list(gt,meshes)
    images = torch.zeros((2,224,224,4))
    
    count = 0
    for j in pred_scene_list:
        
        images[count] = torch.tensor(render_top_view(j))
        verts= j.verts_packed().cpu().numpy()
        faces = j.faces_packed().cpu().numpy()
        temp = trimesh.Trimesh(vertices =verts,faces=faces)
        temp.export('/home/kaly/research/text2scene/results/pred.obj')
        count+=1

        
    
    
    for j in gt_scene_list:
        
        verts= j.verts_packed().cpu().numpy()
        faces = j.faces_packed().cpu().numpy()
        temp = trimesh.Trimesh(vertices =verts,faces=faces)
        temp.export('/home/kaly/research/text2scene/results/gt.obj')
        images[count] = torch.tensor(render_top_view(j))
        count+=1
    #meshes = 

    epoch = torch.load('/home/kaly/research/text2scene/pose_weights/pose_diffusion_epoch_21.pth')['epoch']
    grid = utils.make_grid(images, 2).cpu()
    #filename = f'/home/kaly/research/text2scene/results/demo_{epoch:05}.png'
    #TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
    #display.display(display.Image(filename))
    tqdm.write('')
demo()