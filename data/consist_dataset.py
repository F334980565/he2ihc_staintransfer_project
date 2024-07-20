import torch
import itertools
import os
import pandas as pd
from PIL import Image
import numpy as np
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ConsistDataset(BaseDataset): #这个是consistcycle用的数据集，目的是标准化IHC
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        #self.slice_list = opt.slice_list
        self.predict_path = '/home/s611/Projects/wu/he2ihc_classify_project/probs_save/IHC_all/csv/probs.csv'
        self.target_list = ['A16886']
        self.src_list = ['C105560', 'C104494', 'C113327']
        self.slice_dict = {}
        self.target_index_ranges = {}
        self.src_index_ranges = {}
        
        current_index = 0
        for slice_name in self.target_list:
            slice_file_list = sorted(os.listdir(os.path.join(opt.src_path, slice_name, 'he')))
            slice_he_paths = [os.path.join(opt.src_path, slice_name, 'he', filename) for filename in slice_file_list]
            slice_ihc_paths = [os.path.join(opt.src_path, slice_name, 'ihc', filename) for filename in slice_file_list]
            self.slice_dict[slice_name] = [slice_he_paths, slice_ihc_paths]
            slice_len = len(self.slice_dict[slice_name][0])
            self.target_index_ranges[slice_name] = (current_index, current_index + slice_len)
            current_index += slice_len
        
        current_index = 0
        for slice_name in self.src_list:
            slice_file_list = sorted(os.listdir(os.path.join(opt.src_path, slice_name, 'he')))
            slice_he_paths = [os.path.join(opt.src_path, slice_name, 'he', filename) for filename in slice_file_list]
            slice_ihc_paths = [os.path.join(opt.src_path, slice_name, 'ihc', filename) for filename in slice_file_list]
            self.slice_dict[slice_name] = [slice_he_paths, slice_ihc_paths]
            slice_len = len(self.slice_dict[slice_name][0])
            self.src_index_ranges[slice_name] = (current_index, current_index + slice_len)
            current_index += slice_len
        
        print(self.target_index_ranges)
        print(self.src_index_ranges)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        
        #将ihc patch的分类结果加到数据列表里
        df = pd.read_csv(self.predict_path)
        img_paths = df['img_path'].tolist()
        predicts = df['predict'].tolist()
        predict_dict = dict(zip(img_paths, predicts))
        for slice_name in self.target_list + self.src_list:
            slice_predicts = []
            ihc_paths = self.slice_dict[slice_name][1]
            for ihc_path in ihc_paths:
                slice_predicts.append(predict_dict.get(ihc_path))
            self.slice_dict[slice_name].append(slice_predicts)
            
        #生成一个乱序索引表，按分类结果排
        refer_ihc_paths = []
        refer_ihc_predicts = []
        for slice_name in self.target_list:
            refer_ihc_paths += self.slice_dict[slice_name][1]
            refer_ihc_predicts += self.slice_dict[slice_name][2]
        
        self.refer_paths_0 = []
        self.refer_paths_1 = []
        self.refer_paths_2 = []
        for path, predict in zip(refer_ihc_paths, refer_ihc_predicts):
            if predict == 0:
                self.refer_paths_0.append(path)
            elif predict == 1:
                self.refer_paths_1.append(path)
            elif predict == 2:
                self.refer_paths_2.append(path)
                
        self.refer_iterator_0 = itertools.cycle(self.refer_paths_0)
        self.refer_iterator_1 = itertools.cycle(self.refer_paths_1)
        self.refer_iterator_2 = itertools.cycle(self.refer_paths_2)

    def __getitem__(self, index):
        
        for slice_name, (start, end) in self.src_index_ranges.items(): #
            if start <= index < end:
                relative_index = index - start
                he_path = self.slice_dict[slice_name][0][relative_index]
                ihc_path = self.slice_dict[slice_name][1][relative_index]
                predict = self.slice_dict[slice_name][2][relative_index]
                cur_slice = slice_name
                break
        
        #选取target切片中同一分类结果的图像作为参考图像，用iterator确保遍历
        if predict == 0:
            refer_path = next(self.refer_iterator_0)
        if predict == 1:
            refer_path = next(self.refer_iterator_1)
        if predict == 2:
            refer_path = next(self.refer_iterator_2)
        
        refer_img = Image.open(refer_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')
        
        transform_params = get_params(self.opt, refer_img.size)
        transform = get_transform(self.opt, transform_params)
        
        refer_tensor = transform(refer_img)
        ihc_tensor = transform(ihc_img)

        return {'refer': refer_tensor, 'ihc': ihc_tensor, 'refer_path': refer_path, 'ihc_path': ihc_path, 'slice':cur_slice, 'predict': predict}

    def __len__(self):
        """只计算src的数量"""
        n = max(end for _, (start, end) in self.src_index_ranges.items())
            
        return n