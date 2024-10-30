import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize

from dataset.utils import crop_arrays


class ISIC2018Dataset(Dataset):
    '''
    ISIC2018 dataset
    '''
    def __init__(self, config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1):
        super().__init__()
        self.base_size = base_size
        data_root = config.path_dict[config.dataset]
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.image_dir = os.path.join(data_root, 'original', 'images')
            self.label_dir = os.path.join(data_root, 'original', 'labels')
        else:
            self.image_dir = os.path.join(data_root, f'resized_{base_size[0]}', 'images')
            self.label_dir = os.path.join(data_root, f'resized_{base_size[0]}', 'labels')

        self.img_size = crop_size
        self.resize = resize
        self.shot = config.shot
        self.support_idx = config.support_idx
        self.precision = config.precision

        assert split in ['train', 'valid', 'test']
        self.dset_size = dset_size
        self.base_size = base_size
        self.toten = ToTensor()
        self.resizer = Resize(self.img_size)
        self.resize = resize

        ptf = '' if self.support_idx == 0 else f'_{self.support_idx}'
        if split == 'test':
            file_path = os.path.join(data_root, 'meta', f'test_files{ptf}.pth')
        else:
            file_path = os.path.join(data_root, 'meta', f'train_files{ptf}.pth')
        file_dict = torch.load(file_path)

        self.data_idxs = []
        if split == 'test':
            for class_name in file_dict.keys():
                self.data_idxs += file_dict[class_name]
        else:
            # get class names sorted by number of files
            class_names = list(file_dict.keys())
            class_names.sort(key=lambda x: len(file_dict[x]))
            class_names = list(reversed(class_names))

            # choose number of files per class
            shot_per_class = [self.shot // len(class_names) for _ in range(len(class_names))]
            for i in range(self.shot % len(class_names)):
                shot_per_class[i] += 1
            
            # choose files for training
            if split == 'train':
                for class_name, n_files in zip(class_names, shot_per_class):
                    self.data_idxs += file_dict[class_name][self.support_idx*n_files:(self.support_idx+1)*n_files]
                    if len(file_dict[class_name]) - self.support_idx*n_files < n_files:
                        self.data_idxs += file_dict[class_name][:(n_files - len(file_dict[class_name]) + self.support_idx*n_files)]
            
            # choose files for validation
            else:
                files_per_class = [dset_size // len(class_names) for _ in range(len(class_names))]
                for i in range(dset_size % len(class_names)):
                    files_per_class[i] += 1
        
                for class_name, n_files_train, n_files_val in zip(class_names, shot_per_class, files_per_class):
                    valid_files = file_dict[class_name][:self.support_idx*n_files_train] + file_dict[class_name][(self.support_idx+1)*n_files_train:]
                    self.data_idxs += valid_files[:n_files_val]

        if dset_size < 0:
            self.dset_size = len(self.data_idxs)
        elif not eval_mode:
            self.dset_size = dset_size
        else:
            self.dset_size = min(dset_size, len(self.data_idxs))
            
    def __len__(self):
        return self.dset_size
    
    def __getitem__(self, idx):
        img_path = self.data_idxs[idx % len(self.data_idxs)] + '.jpg'
        image, label = self.load_data(img_path)
        
        return self.postprocess_data(image, label)
    
    def load_data(self, img_path):
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
        lbl_path = img_path.replace('.jpg', '.png')
        label = Image.open(os.path.join(self.label_dir, lbl_path))
    
        return image, label
    
    def postprocess_data(self, image, label):
        if self.eval_mode:
            Y_full = self.toten(label.resize((512, 512)))
            image = image.resize((self.base_size[1], self.base_size[0]))
            label = label.resize((self.base_size[1], self.base_size[0]))

        X = self.toten(image)
        Y = self.toten(label)
            
        if self.precision == 'bf16':
            X = X.to(torch.bfloat16)
            Y = Y.to(torch.bfloat16)

        if self.resize:
            X = self.resizer(X)
            Y = self.resizer(Y)
        else:
            X, Y = crop_arrays(X, Y,
                               base_size=X.size()[-2:],
                               crop_size=self.img_size,
                               random=(not self.eval_mode))
        M = torch.ones_like(Y)
            
        if self.eval_mode:
            return X, Y, M, Y_full
        else:
            return X, Y, M
