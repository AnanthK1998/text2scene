
from torch.utils.data import Dataset,DataLoader
import clip 
import torch
import pickle
import numpy as np
import os
import json



class PoseGen(Dataset):
    def __init__(self, split="train",overfit=False,use_cache=False,inference =False,normalize_text=True):
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
        self.inference = inference
        self.normalize_text = normalize_text
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        a=0 #set to -1 
        b=1
        a2=-1
        b2=1
        text_embeddings = torch.zeros((40,512))
        box3D = torch.zeros((40,9))
        objectness = torch.zeros((40,1))
        data = self.data[index]
        count=0
        # max1 = torch.tensor(7.5061) 
        # min1 = torch.tensor(-7.5617)
        max1 = torch.tensor([[[1.4011],
                 [6.3341],
                 [17.7243],
                 [2.6095],
                 [4.4751],
                 [2.8320],
                 [3.1415],
                 [44]]])
        min1 = torch.tensor([[[-11.1616],
                 [-8.7337],
                 [0.0000],
                 [ 0.0000],
                 [ 0.0000],
                 [ 0.0000],
                 [-3.1416],
                 [0]]])
        
        max2= torch.tensor(7.0983) 
        min2= torch.tensor(-3.4024)
        scene_id = data['scene_id']
        file = open('/home/kaly/research/text2scene/scannet_planes/scannet_planes/'+scene_id+'.json')
        scene_verts = np.array(json.load(file)['verts'])
        scene_verts = scene_verts[scene_verts[:,2].argsort()]
        mid = int(scene_verts.shape[0]/2)
        floor_verts = scene_verts[:mid,:]
        floor_center = np.mean(floor_verts,axis=0)
        
        for i in data['objects']:
            try:
                text_embed = self.clip_model.encode_text(clip.tokenize(i['object_data']['text']))
                pose = torch.tensor(i['object_data']['box3D'])
                text_embeddings[count] = text_embed.detach()
                box3D[count,:7] = pose
                box3D[count,7] = int(i['object_data']['cls_id'])+1
                box3D[count,8] = 1
                count+=1
            except (KeyError,RuntimeError) as e : #sometimes text token context length>77, hence runtime error. skip those
                continue
        

        #calculate relative coordinates wrt floor
        translation = box3D[:,0:3]
        floor_center = torch.ones((40,3))*floor_center
        box3D[:,0:3] = translation-floor_center
        
        if count<40: #PADDING
            objectness[count:40] = 0
            text_embeddings[count:40] = torch.zeros((512))
            box3D[count:40,:7] = torch.zeros((7))
            box3D[count:40,7] = 0 #unknown class
            box3D[count:40,8] = 0 #objectness zero for padded objects
        
        text_embeddings = text_embeddings.permute(1,0)
        box3D = box3D.permute(1,0)
        objectness = objectness.permute(1,0)
        size = count
        box3D[:8,:] = a + ((box3D[:8,:] - min1[0]) * (b - a) / (max1[0]- min1[0])) #min-max norm to [0,1]
        if self.normalize_text:
            text_embeddings = a2 + ((text_embeddings - min2) * (b2 - a2) / (max2 - min2)) #min-max norm to [-1,1]
        
        return text_embeddings,box3D,size
    
    def __len__(self):
        return len(self.data)
    



        #max1 = torch.tensor([[[4.6341],
        #          [7.5061],
        #          [2.1705],
        #          [3.3867],
        #          [5.2126],
        #          [2.8320],
        #          [3.1415]]])
        # min1 = torch.tensor([[[-3.9863],
        #          [-7.5617],
        #          [-0.3567],
        #          [ 0.0000],
        #          [ 0.0000],
        #          [ 0.0000],
        #          [-3.1416]]])