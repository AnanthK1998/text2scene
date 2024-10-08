## text2scene: 3D Scene Generation conditioned on Natural Languages

### Install
1. This implementation uses Python 3.6, [Pytorch1.7.1](http://pytorch.org/), cudatoolkit 11.0. We recommend to use [conda](https://docs.conda.io/en/latest/miniconda.html) to deploy the environment.
   * Install with conda:
    ```
    conda env create -f environment.yml
    conda activate scenegen
    ```
    * Install [PyTorch3D](https://pytorch3d.org/), [CLIP](https://github.com/openai/CLIP)
    ```
    conda install pytorch3d -c pytorch3d
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
    ```

    * Install with pip:
    ```
    pip install -r requirements.txt
    ```

### Prepare Data
In our paper, we use the annotated instance CAD models from the [Scan2CAD](https://github.com/skanti/Scan2CAD) dataset.
Scan2CAD aligns the object CAD models from [ShapeNetCore.v2](https://shapenet.org/) to each object in [ScanNet](http://www.scan-net.org/), and we use these aligned CAD models as the ground-truth for the mesh branch.

##### Preprocess ScanNet and Scan2CAD data
You can either directly download the processed samples [[link](https://tumde-my.sharepoint.com/:u:/g/personal/yinyu_nie_tum_de/EdTtS1JDX35DoZHj11Y5Vb8Bw89ollS_-pxiPGPjqvqZyA?e=H9pe3B)] to the directory below (recommended)
```
datasets/scannet/processed_data/
```
or <br>

1. Ask for the [ScanNet](http://www.scan-net.org/) dataset and download it to
   ```
   datasets/scannet/scans
   ```
2. Ask for the [Scan2CAD](https://github.com/skanti/Scan2CAD) dataset and download it to
   ```
   datasets/scannet/scan2cad_download_link
   ```
3. Preprocess the ScanNet and Scan2CAD dataset for training by
   ```
   cd text2scene
   python utils/scannet/gen_scannet_w_orientation.py
   ```

##### Preprocess ShapeNet data
You can either directly download the processed data [[link](https://tumde-my.sharepoint.com/:u:/g/personal/yinyu_nie_tum_de/EQfn3F28ie9LlM6qq66QuXcBFe4HjCsBZGJtm9eLw9XrhQ?e=NLyZJP)] and extract them to `datasets/ShapeNetv2_data/` as below
```
datasets/ShapeNetv2_data/point
datasets/ShapeNetv2_data/pointcloud
datasets/ShapeNetv2_data/voxel
datasets/ShapeNetv2_data/watertight_scaled_simplified
```
or <br>

1. Download [ShapeNetCore.v2](https://shapenet.org/) to the path below
   
    ```
    datasets/ShapeNetCore.v2
   ```
   
2. Process ShapeNet models into watertight meshes by the following.
   
    ```
    python utils/shapenet/1_fuse_shapenetv2.py
   ```
   
   If it does not work, please delete the `./build` and `.so` file in `external/librender/` and recompile the pyrender by
   
    ```
    cd text2scene/external/librender
    rm -rf ./build ./*.so
    python setup.py build_ext --inplace
   ```
   
3. Sample points on ShapeNet models for training (similar to [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks)).
   
    ```
    python utils/shapenet/2_sample_mesh.py --resize --packbits --float16
   ```
   
4. There are usually 100K+ points per object mesh. We simplify them to speed up our testing and visualization by
   
    ```
    python utils/shapenet/3_simplify_fusion.py --in_dir datasets/ShapeNetv2_data/watertight_scaled --out_dir datasets/ShapeNetv2_data/watertight_scaled_simplified
   ```
   

##### Verify preprocessed data
   After preprocessed the data, you can run the visualization script below to check if they are generated correctly.
   
   * Visualize ScanNet+Scan2CAD+ShapeNet samples by
     
      ```
      python utils/scannet/visualization/vis_gt.py
      ```
     
      A VTK window will be popped up like below.
   
      <img src="out/samples/scene0001_00/verify.png" alt="verify.png" width="60%" />
##### ScanRefer Dataset

If you would like to access to the ScanRefer dataset, please fill out [this form](https://forms.gle/aLtzXN12DsYDMSXX6). Once your request is accepted, you will receive an email with the download link.

> Note: In addition to language annotations in ScanRefer dataset, you also need to access the original ScanNet dataset. Please refer to the [ScanNet Instructions](data/scannet/README.md) for more details.

Download the dataset by simply executing the wget command:
```shell
wget <download_link>
```
Download the ScanRefer dataset and unzip it under `datasets/`. 

##### ScanRefer Data format
```
"scene_id": [ScanNet scene id, e.g. "scene0000_00"],
"object_id": [ScanNet object id (corresponds to "objectId" in ScanNet aggregation file), e.g. "34"],
"object_name": [ScanNet object name (corresponds to "label" in ScanNet aggregation file), e.g. "coffee_table"],
"ann_id": [description id, e.g. "1"],
"description": [...],
"token": [a list of tokens from the tokenized description] 
```
##### Preprocessing the ScanRefer dataset and combining it with above
```
python utils/data_prep.py
```
