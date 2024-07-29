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

class HE2IHCTEST2Dataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.src_path = opt.src_path
        self.slice_list = sorted([d for d in os.listdir(self.src_path) if os.path.isdir(os.path.join(self.src_path, d))])
        
        self.slice_dict = {}
        self.index_ranges = {}
        current_index = 0
        
        for slice_name in self.slice_list:
            he_file_list = os.listdir(os.path.join(opt.src_path, slice_name, 'HE_xiufu'))
            slice_he_paths = [os.path.join(opt.src_path, slice_name, 'HE_xiufu', filename) for filename in he_file_list]
            
            self.slice_dict[slice_name] = [slice_he_paths]
            slice_len = len(self.slice_dict[slice_name][0])
            self.index_ranges[slice_name] = (current_index, current_index + slice_len)
            current_index += slice_len
            print(slice_len)

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        
        for slice_name, (start, end) in self.index_ranges.items():
            if start <= index < end:
                relative_index = index - start
                he_path = self.slice_dict[slice_name][0][relative_index]
                cur_slice = slice_name
                break

        he_img = Image.open(he_path).convert('RGB')
        
        transform_params = get_params(self.opt, he_img.size)
        transform = get_transform(self.opt, transform_params)
        
        he_tensor = transform(he_img)
    
        data = {'A': he_tensor, 'B': he_tensor, 'A_path': he_path, 'B_path': he_path, 'label': 0, 'label_tensor': 0, 'slice':cur_slice}
            
        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        n = 0
        for slice in self.slice_list:
            n += len(self.slice_dict[slice][0])
        print('数据集数量：', n)
            
        return n

    
    