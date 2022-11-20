from torch.utils.data import Dataset
import clip 
import pickle
import torch

class PoseGen(Dataset):
    def __init__(self, split="train",overfit=True):
        self.split = split
        self.overfit = overfit
        if self.split == 'train':
            self.data = pickle.load( open( "/home/kaly/research/RfDNet/datasets/scenegen_train.pkl","rb"))
        elif self.split=='val':
            self.data =pickle.load( open( "/home/kaly/research/RfDNet/datasets/scenegen_val.pkl","rb"))
        if self.overfit:
            self.data = self.data[0:32]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        a=-1
        b=1
        text_embeddings = torch.zeros((40,512))
        box3D = torch.zeros((40,7))
        clip_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=True)
        data = self.data[index]
        count=0
        max1 = torch.tensor([[[3.7593],
                 [7.5061],
                 [2.1705],
                 [3.3867],
                 [4.4751],
                 [2.8320],
                 [3.1415]]])
        min1 = torch.tensor([[[-3.7312],
                 [-7.5617],
                 [-0.3567],
                 [ 0.0000],
                 [ 0.0000],
                 [ 0.0000],
                 [-3.1416]]])
        for i in data['objects']:
            try:
                text_embed = clip_model.encode_text(clip.tokenize(i['object_data']['text']))
                pose = torch.tensor(i['object_data']['box3D'])
                text_embeddings[count] = text_embed.detach()
                box3D[count] = pose
                count+=1
            except:
                continue
                
        if count<40:
            text_embeddings[count:40] = text_embeddings[count-1]
            box3D[count:40] = box3D[count-1]
        text_embeddings = text_embeddings.permute(1,0)
        box3D = box3D.permute(1,0)
        box3D = a + ((box3D - min1[0]) * (b - a) / (max1[0] - min1[0])) #min-max norm to [-1,1]
        
        return text_embeddings,box3D
    
    def __len__(self):
        return len(self.data)
        
        