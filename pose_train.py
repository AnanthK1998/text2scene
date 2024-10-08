from diffusion_model import Unet1D,GaussianDiffusion1D, Trainer

device = 'cuda:0'
model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 9,
    self_condition=True,
    use_dgcnn=False,
    use_stn=True
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
    train_lr = 1e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    use_wandb=True,
    results_folder='./pose_weights_stn/'
)


pretrained = True
if pretrained:
    trainer.load(82)
    
trainer.train()