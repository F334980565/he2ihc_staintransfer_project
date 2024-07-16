import torch
import os
from PIL import Image
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TESTDataset(BaseDataset):
    
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        #如果不成对, 那只要输A_root的参数
        self.dir_A = opt.A_root
        if opt.B_root == None:
            self.dir_B = self.dir_A
        else:
            self.dir_B = opt.B_root
            
        A_list = sorted(os.listdir(self.dir_A))
        B_list = sorted(os.listdir(self.dir_B))

        self.A_paths = [os.path.join(self.dir_A, filename) for filename in A_list]
        self.B_paths = [os.path.join(self.dir_B, filename) for filename in B_list]

        assert(self.opt.load_size >= self.opt.crop_size)  

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        transform_params = get_params(self.opt, (A_img.size))
        transform = get_transform(self.opt, transform_params)
        
        A_tensor = transform(A_img)
        B_tensor = transform(B_img)
        
        return {'A': A_tensor, 'B': B_tensor, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)