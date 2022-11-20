from glob import glob
import json
import numpy as np
import pickle
from easydict import EasyDict
import os
from tqdm import tqdm

CONF = EasyDict()
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/home/kaly/research/RfDNet/"
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



data = []
for i in tqdm(train_scene_list):
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
        with open(CONF.PATH.DATA+"/ScanRefer_filtered_val.json") as json_file:
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
    
    
with open(CONF.PATH.DATA + '/scenegen_val.pkl', 'wb') as f:
    pickle.dump(data, f)