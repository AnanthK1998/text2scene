
from torch.utils.data import Dataset,DataLoader
import clip 
import torch
import pickle
import pickle
import os

class PoseGen(Dataset):
    def __init__(self, split="train",overfit=False,use_cache=False):
        self.split = split
        self.overfit = overfit
        if self.split == 'train':
            self.data = pickle.load( open( "/home/kaly/research/RfDNet/datasets/scenegen_train.pkl","rb"))
        elif self.split=='val':
            self.data =pickle.load( open( "/home/kaly/research/RfDNet/datasets/scenegen_val.pkl","rb"))
        if self.overfit:
            self.data = self.data[0:1]
        self.clip_model,_ = clip.load('ViT-B/32', 'cpu', jit=True)
        self.use_cache = use_cache
        self.cached_data = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        a=0 #set to -1 
        b=1
        a2=-1
        b2=1
        text_embeddings = torch.zeros((40,512))
        box3D = torch.zeros((40,7))
        objectness = torch.zeros((40,1))
        data = self.data[index]
        count=0
        # max1 = torch.tensor(7.5061) 
        # min1 = torch.tensor(-7.5617)
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
        
        for i in data['objects']:
            try:
                text_embed = self.clip_model.encode_text(clip.tokenize(i['object_data']['text']))
                pose = torch.tensor(i['object_data']['box3D'])
                text_embeddings[count] = text_embed.detach()
                box3D[count] = pose
                objectness[count] = 1
                count+=1
            except (KeyError,RuntimeError) as e : #sometimes text token context length>77, hence runtime error. skip those
                continue
        if count<40: #PADDING
            objectness[count:40] = -1
            text_embeddings[count:40] = text_embeddings[count-1]
            box3D[count:40] = box3D[count-1]
            
        text_embeddings = text_embeddings.permute(1,0)
        box3D = box3D.permute(1,0)
        objectness = objectness.permute(1,0)
        box3D = a + ((box3D - min1[0]) * (b - a) / (max1[0] - min1[0])) #min-max norm to [-1,1]
        text_embeddings = a2 + ((text_embeddings - min2) * (b2 - a2) / (max2 - min2))
        # rotation =box3D[:3,:]
        # translation = box3D[3:6,:]
        # orientation = box3D[6,:]
        #del box3D
        return text_embeddings,box3D#rotation,translation,orientation,objectness
    
    def __len__(self):
        return len(self.data)