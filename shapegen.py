from torch.utils.data import Dataset,DataLoader
import clip 
import torch
import pickle
import os
from utils.binvox import read_as_3d_array
from utils.shapenet import ShapeNetv2_path

class ShapeGen(Dataset):
    def __init__(self, split="train",overfit=False):
        self.split = split
        self.overfit = overfit
        if self.split == 'train':
            self.data = pickle.load( open( "/home/kaly/research/RfDNet/datasets/shapegen_train.pkl","rb"))
        elif self.split=='val':
            self.data =pickle.load( open( "/home/kaly/research/RfDNet/datasets/shapgen_val.pkl","rb"))
        if self.overfit:
            self.data = self.data[0:32]
        self.clip_model,_ = clip.load('ViT-B/32', 'cpu', jit=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        a=-1
        b=1
        text_embeddings = torch.zeros((1,512))
        #voxels = torch.zeros((128,128,128))
        
        data = self.data[index]
        
        
        
        max2= torch.tensor(7.0983) 
        min2= torch.tensor(-3.0967)
    
  
        text_embeddings = self.clip_model.encode_text(clip.tokenize(data['text'])).detach()

        
        shapenet_model = os.path.join(ShapeNetv2_path, data['shapenet_catid'], data['shapenet_id'],'models/model_normalized.solid.binvox')
        
        with open(shapenet_model, 'rb') as f:
            voxels = torch.tensor(read_as_3d_array(f).data,dtype=torch.uint8)
            
        
        
        
        text_embeddings = text_embeddings.permute(1,0)
        text_embeddings = a + ((text_embeddings - min2) * (b - a) / (max2 - min2))
        
        return text_embeddings, voxels
    
    def __len__(self):
        return len(self.data)