import sys
import os
import argparse
import math
import urllib.request
import subprocess
from pathlib import Path
from tqdm import tqdm

TRAIN_DATASETS = ['taskonomy', 'coco', 'midair', 'mpii', 'deepfashion', 'freihand']
DOWNSTREAM_DATASETS = ['ap10k', 'davis2017', 'linemod', 'isic2018', 'fsc147', 'cellpose']

DOWNLOAD_URLS = {
    # downstream
    'cellpose': 'https://www.dropbox.com/scl/fi/gy6vde2du1itez3uwbtgg/cellpose.zip?rlkey=0s9t6dr59jpphmxy6em6npk90&st=f9cn87k7&dl=0',
    'fsc147': 'https://www.dropbox.com/scl/fi/7kgiqe8xvlmwqspupnnym/FSC147.zip?rlkey=51y7f6rbp8m6u1pwycstno8m2&st=kqmiyh63&dl=0',
    'isic2018': 'https://www.dropbox.com/scl/fi/u5ju23tuxwrw2ncynq7a6/isic2018.zip?rlkey=ntgpoaqyshuuynzcg91j14yg8&st=8y59arcj&dl=0',
    'linemod': 'https://www.dropbox.com/scl/fi/bbjxmew97dvw5ktvt16w6/linemod.zip?rlkey=mz04dic5w6fzhtnp9bou0nty8&st=zpbi6cfq&dl=0',
    'ap10k': 'https://www.dropbox.com/scl/fi/5ey23if92gkrckxtjbk6d/ap10k.zip?rlkey=6e8eit9u62q5eeq3oyo22n7sn&st=8dt54u5z&dl=0',
    'davis2017': 'https://www.dropbox.com/scl/fi/84699n5odmim6te180hph/DAVIS2017.zip?rlkey=fpxefb96n2q5f2q3asii6tpcw&st=pjb674hv&dl=0',

    # meta-train
    'mpii': 'https://www.dropbox.com/scl/fi/bo0f7qzh84e0qzorkk4tj/mpii.zip?rlkey=bvovex3pn0wv7uxqte82cq5ak&st=kyq2wdm5&dl=0',
    'taskonomy': 'https://www.dropbox.com/scl/fi/nenvc88vo1udxnebyuaie/taskonomy.zip?rlkey=ifhcngtw934fbx16jljctvlad&st=9s7j25rs&dl=0',
    'coco': 'https://www.dropbox.com/scl/fi/qol9c978x273kspdfwxwp/COCO.zip?rlkey=sdpylsqy1eerfidatfhmnub0z&st=xvw2lp8c&dl=0',
    'deepfashion': 'https://www.dropbox.com/scl/fi/7swyo6hanwfntbrbbdxwn/deepfashion.zip?rlkey=jar3gd71rsc76d807qwp336tg&st=k14n0noz&dl=0',
    'freihand': 'https://www.dropbox.com/scl/fi/prb7ehhghd63xn7oomo7k/freihand.zip?rlkey=iuea78s468obgzvuulfdfhhjr&st=9ac3hz21&dl=0',
    'midair': 'https://www.dropbox.com/scl/fi/88b991e93ezqi1yq3knlm/MidAir.zip\?rlkey\=pa0dwjyotzsdkm9klbxnl20mb\&st\=kwrrdr0a\&dl\=0',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, default=False)
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'downstream', 'train'] + TRAIN_DATASETS + DOWNSTREAM_DATASETS) 
    args = parser.parse_args()
    
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.mode == 'all':
        dataset_list = TRAIN_DATASETS + DOWNSTREAM_DATASETS
    elif args.mode == 'downstream':
        dataset_list = DOWNSTREAM_DATASETS
    elif args.mode == 'train':
        dataset_list = TRAIN_DATASETS
    else:
        dataset_list = [args.mode]
    
    data_paths = { }
    for dataset in dataset_list:
        url = DOWNLOAD_URLS[dataset]
        out_dir = str(root / f'{dataset}.zip')
        if url is None:
            print(f'Currently {dataset} is not supported')
        else:
            print(f'Download {dataset} from {url}')
        
        subprocess.call(f'wget {url} -O {out_dir}'.split())

        # write data paths to data_paths.yaml
