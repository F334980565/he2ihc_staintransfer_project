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

class HE2IHCtempDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.slice_list = opt.slice_list
        self.src_path = opt.src_path
        self.label_path = opt.label_path #如果有就按label_path里.csv保存的label, 没有就用pool的方式生成label, 至于用不用在于模型。
        
        if self.slice_list == ['all']:
            self.slice_list = sorted([d for d in os.listdir(self.src_path) if os.path.isdir(os.path.join(self.src_path, d))])
        self.slice_dict = {}
        self.index_ranges = {}
        self.map = {0: 1, 1: -1, 2: 0} #思来想去作为label还是该给这种值吧？woc 最后用的是哪个？？？？
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
        
        if not self.label_path == None:
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

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        
        for slice_name, (start, end) in self.index_ranges.items():
            if start <= index < end:
                relative_index = index - start
                he_path = self.slice_dict[slice_name][0][relative_index]
                ihc_path = self.slice_dict[slice_name][1][relative_index]
                if not self.label_path == None:
                    label = self.slice_dict[slice_name][2][relative_index]
                    label = self.map.get(label)
                cur_slice = slice_name
                break

        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')
        
        transform_params = get_params(self.opt, he_img.size)
        transform = get_transform(self.opt, transform_params)
        
        he_tensor = transform(he_img)
        ihc_tensor = transform(ihc_img)
        
        if self.label_path == None:
            label, label_tensor = self.get_label_tensor(ihc_tensor, params = [8, 2.25, 0.7])
        else:
            _, label_tensor = self.get_label_tensor(ihc_tensor, params = [8, 2.25, 0.7])
    
        data = {'A': he_tensor, 'B': ihc_tensor, 'A_path': he_path, 'B_path': ihc_path, 'label': label, 'label_tensor': label_tensor, 'slice':cur_slice}
            
        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        n = 0
        for slice in self.slice_list:
            n += len(self.slice_dict[slice][0])
        print('数据集数量：', n)
            
        return n
    
    def get_label_tensor(self, img_tensor, params = [8, 2.25, 0.7]):
        
        img_tensor = (1 - img_tensor) / 2
        img_size = img_tensor.shape[-1]
        k, positive_threshold, background_threshold = params
        avg_pool_8 = F.avg_pool2d(img_tensor, kernel_size=(k, k))
        max_pool_256 = F.max_pool2d(avg_pool_8, kernel_size=(img_size // (2*k), img_size // (2*k)))
        max_pool_512 = F.max_pool2d(avg_pool_8, kernel_size=(img_size // k, img_size // k))
        sum = 0.0 * max_pool_512[0] + 0.0 * max_pool_512[1] + 3.0 * max_pool_512[2]
        if sum > positive_threshold: #
            label = 1
        elif sum <= positive_threshold and sum >= background_threshold:
            label = -1
        else:
            label = 0
        
        summed_tensor = torch.sum(max_pool_256, dim=0)
        # 如果我没想到怎么用分类的方法搞，还是智能L1的话，那就改成1,0,-1模式的
        label_tensor = torch.where(summed_tensor > positive_threshold, 1,
                                    torch.where((summed_tensor <= positive_threshold) & (summed_tensor >= background_threshold), -1, 0))
        
        return label, label_tensor

    
    