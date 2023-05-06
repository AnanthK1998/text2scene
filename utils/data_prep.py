from glob import glob
import json
import numpy as np
import pickle
from easydict import EasyDict
import os
from tqdm import tqdm
import clip

CONF = EasyDict()
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/home/kaly/research/RfDNet/" #add path to the project directory
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "datasets")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet/scannet")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")
CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 126
CONF.TRAIN.SEED = 42

device= "cuda:0"

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

train_scene_list = sorted([line.rstrip() for line in open(CONF.PATH.DATA+'/ScanRefer_filtered_train.txt')])
val_scene_list = sorted([line.rstrip() for line in open(CONF.PATH.DATA+'/ScanRefer_filtered_val.txt')])

scene_dir = glob(CONF.PATH.DATA+ "/scannet/processed_data/*")

clip_model,_ = clip.load('ViT-B/32', 'cpu', jit=True)

data = []
for i in tqdm(train_scene_list):
    try:
        rfd_data = pickle.load( open( CONF.PATH.DATA+ "/scannet/processed_data/"+ i+ "/bbox.pkl","rb"))
    except:
        continue
    instance_ids=[]
    for obj in rfd_data:
        instance_ids.append(int(obj['instance_id']))
    scene_dict = {}
    scene_dict['scene_id'] = i
    scene_dict['objects'] = []
    scene_dict1 = {}
    scene_dict1['scene_id'] = i
    scene_dict1['objects'] = []
    scene_dict2 = {}
    scene_dict2['scene_id'] = i
    scene_dict2['objects'] = []
    scene_dict3 = {}
    scene_dict3['scene_id'] = i
    scene_dict3['objects'] = []
    scene_dict4 = {}
    scene_dict4['scene_id'] = i
    scene_dict4['objects'] = []

    for object in instance_ids:
        object_dict = {}
        object_dict['object_id'] = object
        object_dict['object_data'] = {}
        object_dict1 = {}
        object_dict1['object_id'] = object
        object_dict1['object_data'] = {}
        object_dict2 = {}
        object_dict2['object_id'] = object
        object_dict2['object_data'] = {}
        object_dict3 = {}
        object_dict3['object_id'] = object
        object_dict3['object_data'] = {}
        object_dict3 = {}
        object_dict3['object_id'] = object
        object_dict3['object_data'] = {}
        object_dict4 = {}
        object_dict4['object_id'] = object
        object_dict4['object_data'] = {}
        with open(CONF.PATH.DATA+"/ScanRefer_filtered_train.json") as json_file:
            text_data = json.load(json_file)
        for text_dict in text_data:
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='0':
                object_dict['object_data']['text'] = text_dict['description']
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='1':
                object_dict1['object_data']['text'] = text_dict['description']
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='2':
                object_dict2['object_data']['text'] = text_dict['description']
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='3':
                object_dict3['object_data']['text'] = text_dict['description']
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='4':
                object_dict4['object_data']['text'] = text_dict['description']
        for pose_dict in rfd_data:
            if pose_dict['instance_id'] == object:
                object_dict['object_data']['box3D'] = pose_dict['box3D']
                object_dict['object_data']['cls_id'] = pose_dict['cls_id']
                object_dict['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict['object_data']['shapenet_id'] = pose_dict['shapenet_id']
                object_dict1['object_data']['box3D'] = pose_dict['box3D']
                object_dict1['object_data']['cls_id'] = pose_dict['cls_id']
                object_dict1['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict1['object_data']['shapenet_id'] = pose_dict['shapenet_id']
                object_dict2['object_data']['box3D'] = pose_dict['box3D']
                object_dict2['object_data']['cls_id'] = pose_dict['cls_id']
                object_dict2['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict2['object_data']['shapenet_id'] = pose_dict['shapenet_id']
                object_dict3['object_data']['box3D'] = pose_dict['box3D']
                object_dict3['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict3['object_data']['shapenet_id'] = pose_dict['shapenet_id']
                object_dict4['object_data']['box3D'] = pose_dict['box3D']
                object_dict4['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict4['object_data']['shapenet_id'] = pose_dict['shapenet_id']
        
        scene_dict['objects'].append(object_dict)
        scene_dict1['objects'].append(object_dict1)
        scene_dict2['objects'].append(object_dict2)
        scene_dict3['objects'].append(object_dict3)
        scene_dict4['objects'].append(object_dict4)
    data.append(scene_dict)
    data.append(scene_dict1)
    data.append(scene_dict2)
    data.append(scene_dict3)
    data.append(scene_dict4)

    

for i in tqdm(val_scene_list):
    try:
        rfd_data = rfd_data = pickle.load( open( CONF.PATH.DATA+ "/scannet/processed_data/"+ i+ "/bbox.pkl","rb"))
    except:
        continue
    instance_ids=[]
    for obj in rfd_data:
        instance_ids.append(int(obj['instance_id']))
    
    scene_dict1 = {}
    scene_dict1['scene_id'] = i
    scene_dict1['objects'] = []
    scene_dict2 = {}
    scene_dict2['scene_id'] = i
    scene_dict2['objects'] = []
    scene_dict3 = {}
    scene_dict3['scene_id'] = i
    scene_dict3['objects'] = []
    scene_dict4 = {}
    scene_dict4['scene_id'] = i
    scene_dict4['objects'] = []
    
    for object in instance_ids:
        object_dict1 = {}
        object_dict1['object_id'] = object
        object_dict1['object_data'] = {}
        object_dict2 = {}
        object_dict2['object_id'] = object
        object_dict2['object_data'] = {}
        object_dict3 = {}
        object_dict3['object_id'] = object
        object_dict3['object_data'] = {}
        object_dict3 = {}
        object_dict3['object_id'] = object
        object_dict3['object_data'] = {}
        object_dict4 = {}
        object_dict4['object_id'] = object
        object_dict4['object_data'] = {}
        with open(CONF.PATH.DATA+"/ScanRefer_filtered_val.json") as json_file:
            text_data = json.load(json_file)
        for text_dict in text_data:

            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='1':
                object_dict1['object_data']['text'] = text_dict['description']
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='2':
                object_dict2['object_data']['text'] = text_dict['description']
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='3':
                object_dict3['object_data']['text'] = text_dict['description']
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='4':
                object_dict4['object_data']['text'] = text_dict['description']
        for pose_dict in rfd_data:
            if pose_dict['instance_id'] == object:
                object_dict1['object_data']['box3D'] = pose_dict['box3D']
                object_dict1['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict1['object_data']['shapenet_id'] = pose_dict['shapenet_id']
                object_dict2['object_data']['box3D'] = pose_dict['box3D']
                object_dict2['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict2['object_data']['shapenet_id'] = pose_dict['shapenet_id']
                object_dict3['object_data']['box3D'] = pose_dict['box3D']
                object_dict3['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict3['object_data']['shapenet_id'] = pose_dict['shapenet_id']
                object_dict4['object_data']['box3D'] = pose_dict['box3D']
                object_dict4['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict4['object_data']['shapenet_id'] = pose_dict['shapenet_id']
        
        scene_dict1['objects'].append(object_dict1)
        scene_dict2['objects'].append(object_dict2)
        scene_dict3['objects'].append(object_dict3)
        scene_dict4['objects'].append(object_dict4)
    
    data.append(scene_dict1)
    data.append(scene_dict2)
    data.append(scene_dict3)
    data.append(scene_dict4)

## ELIMINATE SINGLE OBJECT SCENES, CAUSES ERROR IN DATALOADER OTHERWISE
#removing null scenes    
l=[295,296,297,298,299,670,671,672,673,674,745,746,747,748,749,910,911,912,913,914,990,1027,1028,1270,1271,1272,1274,1319,1696,1820,1821,1822,1823,1824,2916,2917,3006,3007,3008,3009] 
l.reverse()

for idx in l:
    del data[idx]

#Remove scenes with no floor layouts available
l = np.load('/home/kaly/research/text2scene/datasets/null_scenecad.npy').tolist()
l.reverse()
for idx in l:
    del data[idx]

with open(CONF.PATH.DATA + '/scenegen_train.pkl', 'wb') as fp:
    pickle.dump(data, fp)



#Validation
data = []
for i in tqdm(val_scene_list):
    try:
        rfd_data = rfd_data = pickle.load( open( CONF.PATH.DATA+ "/scannet/processed_data/"+ i+ "/bbox.pkl","rb"))
    except:
        continue
    instance_ids=[]
    for obj in rfd_data:
        instance_ids.append(int(obj['instance_id']))
    scene_dict = {}
    scene_dict['scene_id'] = i
    scene_dict['objects'] = []
    
    for object in instance_ids:
        object_dict = {}
        object_dict['object_id'] = object
        object_dict['object_data'] = {}
        with open("/home/kaly/research/RfDNet/datasets/ScanRefer_filtered_val.json") as json_file:
            text_data = json.load(json_file)
        for text_dict in text_data:
            if text_dict['scene_id']==i and text_dict['object_id'] == str(object) and text_dict['ann_id']=='0':
                object_dict['object_data']['text'] = text_dict['description']
        for pose_dict in rfd_data:
            if pose_dict['instance_id'] == object:
                object_dict['object_data']['box3D'] = pose_dict['box3D']
                object_dict['object_data']['shapenet_catid'] = pose_dict['shapenet_catid']
                object_dict['object_data']['shapenet_id'] = pose_dict['shapenet_id']
        
        scene_dict['objects'].append(object_dict)
    
    data.append(scene_dict)
    

val_l = [59]
val_l.reverse()

for idx in val_l:
    del data[idx]

#Remove scenes with no layout CADS
val_l = np.load('/home/kaly/research/text2scene/datasets/null_scenecad_val.npy').tolist()
val_l.reverse()
for idx in val_l:
    del data[idx]

    
with open(CONF.PATH.DATA + '/scenegen_val.pkl', 'wb') as f:
    pickle.dump(data, f)


