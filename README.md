# Few-shot Generalist for Non-Linguistic Data
This repository is designed for implementing training, finetuning, and testing pipeline of ['Chameleon: A Data-Efficient Generalist for Dense Visual Prediction in the Wild'](https://arxiv.org/pdf/2404.18459) using Gaudi-v2.

## Dependencies
- Install the necessary libraries:
```
$ pip install -r requirements.txt
```

## Setup
- Download meta-training and downstream datasets (will be released soon)

- To download BEITv2-LARGE checkpoint used in meta-training, run the following command:
```
$ python get_beitv2.py
``` 

- Then, create a data_paths.yaml:
```
{dataset1}: {path1}
{dataset2}: {path2}
...
{datasetN}: {pathN}
```

- We provide a meta-trained checkpoint of Chameleon in [this link](https://www.dropbox.com/scl/fi/2ta5kez62nw4lgb8bhvf0/step-400000.pth?rlkey=xilbtrawhua95bb5abywthrrz&st=a6m13x94&dl=0). Please locate it in 'experiments/TRAIN/VTMv2/checkpoints/'. 

## How to Run 
- Meta-Training
```
$ python main.py --stage 0

# Preprocess deepspeed checkpoints (.ckpt) into single file (.pth) after meta-training.  
$ python preprocess_checkpoints.py
```

- Fine-Tuning
```
# DAVIS2017 
$ bash scripts/davis2017/finetune.sh VTMv2 ${CLASSNAME} ${learning_rate} -nd 1 

# FSC147 
$ bash scripts/fsc147/finetune.sh VTMv2 -nd 4

# ISIC2018 (SUPPORT_IDX \in [0,1,2,3,4])
$ bash scripts/isic2018/finetune.sh VTMv2 ${SUPPORT_IDX} -nd 1

# CellPose
$ bash scripts/cellpose/finetune.sh VTMv2 -nd 2

# AP10K
$ bash scripts/ap10k/finetune.sh VTMv2 ${CLASSNAME} -nd 4

# LINEMOD
$ bash scripts/linemod/finetune_segment.sh VTMv2 ${CLASSNAME} -nd 2 -salpha False
$ bash scripts/linemod/finetune_pose.sh VTMv2 ${CLASSNAME} -nd 2 -salpha False
```

## Current Issues (will be fixed)
- Low HPU utilization in dynamic mode.
- Lazy mode is not working.
- The 'separate_alpha' option is not working in dynamic-mode. This may affect the performance of pose estimation in linemod dataset.
