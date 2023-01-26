from diffusion.imagen_pytorch import Imagen,Unet
from shapegen import ShapeGen
from diffusion.imagen_trainer import ImagenTrainer
import wandb
wandb.init(project="shapegen", entity="ananthk")


unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    channels=128,
    channels_out=128,
    layer_attns = (False, True, True, True),
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)
  
  # imagen, which contains the unet above
  
imagen = Imagen(
    condition_on_text = True,  # this must be set to False for unconditional Imagen
    text_embed_dim=512,
    unets = (unet1,unet2),
    channels= 128,
    image_sizes = (64,128),
    timesteps = 1000,
)
  
trainer = ImagenTrainer(
    imagen,
    fp16 = True,
    split_valid_from_train = False,
).cuda()
  
  # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training
  
train_ds = ShapeGen('train')

trainer.add_train_dataset(train_ds, batch_size = 32)
val_ds = ShapeGen('val')

trainer.add_valid_dataset(val_ds, batch_size = 32)

# working training loop
best_loss = 1.
for i in range(100000):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 8)
    if i%10==0:
        print(f'Step: {i}')
        print(f'loss: {loss}')
    wandb.log({'train_loss_unet1':loss})
    if i%1000==0:
        val_loss = trainer.valid_step(unet_number=1,max_batch_size=8)
        print(f'val_loss: {val_loss}')
        if val_loss<best_loss:
            best_loss= val_loss
            trainer.save('shape_weights/best_weight.pt')
        wandb.log({'val_loss_unet1':val_loss})    
# best_loss = 1.
# trainer.load('/home/kaly/research/text2scene/shape_weights/best_weight.pt')
# for i in range(100000):
#     loss = trainer.train_step(unet_number = 2, max_batch_size = 4)
#     if i%10==0:
#         print(f'Step: {i}')
#         print(f'loss: {loss}')
#     wandb.log({'train_loss_unet2':loss})
#     if i%1000==0:
#         val_loss = trainer.valid_step(unet_number=2,max_batch_size=4)
#         print(f'val_loss: {val_loss}')
#         if val_loss<best_loss:
#             best_loss= val_loss
#             trainer.save('shape_weights/best_weight.pt')
#         wandb.log({'val_loss_unet2':val_loss})
    # loss = trainer.train_step(unet_number = 2, max_batch_size = 4)
    # if i%1000==0:
    #     val_loss = trainer.valid_step(unet_number=2,max_batch_size=4)
    #     print(f'val_loss: {val_loss}')
    # print(f'loss: {loss}')

#trainer.save('.shape_weights/best_weight.pt')

#trainer.load('./save.pt')