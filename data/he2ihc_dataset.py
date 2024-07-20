import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import random
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class HE2IHCDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.slice_list = opt.slice_list
        self.src_path = opt.src_path
        self.use_label = opt.use_label
        if self.slice_list == ['all']:
            self.slice_list = sorted([d for d in os.listdir(self.src_path) if os.path.isdir(os.path.join(self.src_path, d))])
        self.slice_dict = {}
        self.index_ranges = {}
        current_index = 0
        
        for slice_name in self.slice_list:
            if opt.use_train_list:
                df = pd.read_csv((os.path.join(opt.src_path, slice_name, 'train_list.csv')))
                slice_file_list = sorted(df['File Name'])
            else:
                he_file_list = os.listdir(os.path.join(opt.src_path, slice_name, 'he'))
                ihc_file_list = os.listdir(os.path.join(opt.src_path, slice_name, 'ihc'))
                slice_file_list = sorted(list(set(he_file_list) & set(ihc_file_list)))
            slice_he_paths = [os.path.join(opt.src_path, slice_name, 'he', filename) for filename in slice_file_list]
            slice_ihc_paths = [os.path.join(opt.src_path, slice_name, 'ihc', filename) for filename in slice_file_list]
            
            self.slice_dict[slice_name] = [slice_he_paths, slice_ihc_paths]
            slice_len = len(self.slice_dict[slice_name][0])
            self.index_ranges[slice_name] = (current_index, current_index + slice_len)
            current_index += slice_len
            print(slice_len)
        
        """
        if self.use_predict:
            df = pd.read_csv(opt.predict_path)
            img_paths = df['img_path'].tolist()
            predicts = df['predict'].tolist()
            predict_dict = dict(zip(img_paths, predicts))
        
            for slice_name in self.slice_list:
                slice_predicts = []
                he_paths = self.slice_dict[slice_name][0]
                ihc_paths = self.slice_dict[slice_name][1]
                for he_path in he_paths:
                    if predict_dict.get(he_path) == None:
                        slice_predicts.append(-1)
                    else:
                        slice_predicts.append(predict_dict.get(he_path))
                self.slice_dict[slice_name].append(slice_predicts)
        """

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        
        for slice_name, (start, end) in self.index_ranges.items():
            if start <= index < end:
                relative_index = index - start
                he_path = self.slice_dict[slice_name][0][relative_index]
                ihc_path = self.slice_dict[slice_name][1][relative_index]
                cur_slice = slice_name
                break

        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')
        
        transform_params = get_params(self.opt, he_img.size)
        transform = get_transform(self.opt, transform_params)
        
        he_tensor = transform(he_img)
        ihc_tensor = transform(ihc_img)
        
        if self.use_label:
            label, label_tensor = self.get_label_tensor(ihc_tensor, params = [1, 1, 1, 1.85, 0.5])
    
            data = {'A': he_tensor, 'B': ihc_tensor, 'A_path': he_path, 'B_path': ihc_path, 'label': label, 'label_tensor': label_tensor, 'slice':cur_slice}
        else:
            data = {'A': he_tensor, 'B': ihc_tensor, 'A_path': he_path, 'B_path': ihc_path, 'slice':cur_slice}
            
        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        n = 0
        for slice in self.slice_list:
            n += len(self.slice_dict[slice][0])
            
        return n
    
    def get_label_tensor(self, img_tensor, params = [1, 1, 1, 1.85, 0.5]):
        img_tensor = (1 - img_tensor) / 2
        a, b, c, positive_threshold, background_threshold = params
        avg_pool_8 = F.avg_pool2d(img_tensor, kernel_size=(8, 8))
        max_pool_256 = F.max_pool2d(avg_pool_8, kernel_size=(32, 32))
        max_pool_512 = F.max_pool2d(avg_pool_8, kernel_size=(64, 64))
        sum = max_pool_512.sum()
        if sum > positive_threshold:
            label = 0
        elif sum <= positive_threshold and sum >= background_threshold:
            label = 1
        else:
            label = 2
        
        summed_tensor = torch.sum(max_pool_256, dim=0)
        # 如果我没想到怎么用分类的方法搞，还是智能L1的话，那就改成1,0,-1模式的
        label_tensor = torch.where(summed_tensor > positive_threshold, 0,
                                    torch.where((summed_tensor <= positive_threshold) & (summed_tensor >= background_threshold), 1, 2))
        
        return label, label_tensor

    
    