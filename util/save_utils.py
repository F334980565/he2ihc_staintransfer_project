"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

def tensor2img(input_image, invert=False): #我觉得he不该反色，但反都反了，不如一路反下去
    
    input_image = input_image[0].cpu()
    scale = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    bias = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    img = input_image * scale + bias
    if invert:
        img = 1 - img
    img = transforms.ToPILImage()(img)
    return img

def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
            
def print_epoch_losses(self, epoch, iters, losses):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- 这个epoch梯度更新的总次数
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
    """
    message = '(epoch: %d, iters: %d) ' % (epoch, iters)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message
    with open(self.log_name_epoch, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message
        
def save_current_results(self, visuals, epoch, name=None): #这个是train里存图的
    """Display current results on visdom; save current results to an HTML file.

    Parameters:
        visuals (OrderedDict) - - dictionary of images to display or save
        epoch (int) - - the current epoch
        save_result (bool) - - if save the current results to an HTML file
    """
    
    for label, image in visuals.items():
        
        img = tensor2img(image, invert=True)
        img_path = os.path.join(self.img_dir, '%s_epoch%.3d_%s.png' % (name, epoch, label))
        img.save(img_path) 

def save_images(visuals, image_path, save_path): #这个是test里存图的

    short_path = os.path.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_data in visuals.items():
        img = tensor2img(im_data, invert=True)
        img.save(os.path.join(save_path, f'{name}_{label}.png'))