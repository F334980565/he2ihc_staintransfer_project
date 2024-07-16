import torch
import torch.nn.functional as F
import itertools
import numpy as np
from PIL import Image
from collections import OrderedDict
from util.image_pool import ImagePool
import torchvision.transforms as transforms
from .base_model import BaseModel
from . import networks
import torchvision.models as models
import lpips
import os

class HE2IHCFLModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
            parser.add_argument('--lambda_identity', type=float, default=0.1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_lpips', type=float, default=5.0, help='weight for VGGlpips loss')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--lambda_T', type=float, default=1.0, help='weight for T loss')
            parser.add_argument('--lambda_discret', type=float, default=5.0, help='weight for discret loss')
            parser.add_argument('--lr_G_A', type=float, default=0.0002, help='learning rate for G_A')
            parser.add_argument('--lr_G_B', type=float, default=0.0002, help='learning rate for G_B')
            parser.add_argument('--lr_D_A', type=float, default=0.0002, help='learning rate for D_A')
            parser.add_argument('--lr_D_B', type=float, default=0.0002, help='learning rate for D_B')
            
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'G_GAN', 'G_lpips', 'G_L1', 'G_T', 'G_discret'] #加了个G_consist
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        self.visual_names = ['real_A', 'real_B', 'fake_B']  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']
            
        self.i = 0
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc + 1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionT = networks.TLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionLpips = lpips.LPIPS(net='vgg', eval_mode=True).eval().requires_grad_(False).to(self.device)
            #self.criterionConsist = torch.nn.CrossEntropyLoss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_A = torch.optim.AdamW(self.netG_A.parameters(), lr=opt.lr_G_A, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.AdamW(self.netD_A.parameters(), lr=opt.lr_D_A, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_D_A)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.condition = input['predict_tensor'].to(self.device)
        bn = self.real_A.shape[0]
        size = self.real_A.shape[1]
        self.real_condition = torch.zeros(self.real_A.shape[0], 1, size, size)
        self.real_condition[:, :, :size//2, :size//2] = self.condition[:, 0, 0].view(bn, 1, size//2, size//2)  # 左上角块
        self.real_condition[:, :, size//2:, :size//2] = self.condition[:, 1, 0].view(bn, 1, size//2, size//2)  # 右上角块
        self.real_condition[:, :, :size//2, size//2:] = self.condition[:, 0, 1].view(bn, 1, size//2, size//2)  # 左下角块
        self.real_condition[:, :, size//2:, size//2:] = self.condition[:, 1, 1].view(bn, 1, size//2, size//2)  # 右下角块
        self.predict_condition = self.real_condition #理论上得用预测模块的输出，但这里还没测试完。
        self.real_A_conditioned = torch.cat(self.real_A, self.real_condition)
        self.real_B_conditioned = torch.cat(self.real_B, self.real_condition)
        
        self.image_paths = input['A_path']
        self.slice_name = input['slice']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A_conditioned)  # G_A(A)
        _, self.fake_condition = self.get_label_tensor(self.fake_B)
        self.fake_B_conditioned = torch.cat(self.real_A, self.real_condition) #这里用fake_condition还是real_condition...我觉得先试试real_condition

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        #real = torch.cat((source, real), 1)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        fake = fake.detach()
        #fake = torch.cat((source, fake), 1)
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake)
        loss_D.backward()
        return loss_D

    def backward_D_A(self): #改动过的
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B)
        #self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_conditioned, self.fake_B_conditioned)

    def backward_G_A(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A

        # GAN loss D_A(G_A(A))
        self.loss_G_GAN = self.criterionGAN(self.netD_A(self.fake_B_conditioned), True)
        self.loss_G_lpips = self.criterionLpips(self.fake_B, self.real_B).mean() * self.opt.lambda_lpips
        # 把改动的L1全加上去
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_T, self.loss_G_discret = self.criterionL1(self.real_B, self.fake_B)
        # 加个对应的consistency loss

        self.loss_G_A = (self.loss_G_GAN*self.opt.lambda_GAN + self.loss_G_lpips_A*self.opt.lambda_lpips + 
                         self.loss_G_L1*self.opt.lambda_L1 + self.loss_G_T*self.opt.lambda_T + self.loss_G_discret*self.opt.lambda_discret)
                        
        # combined loss and calculate gradients
        self.loss_G_A.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netD_A, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G_A.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G_A()             # calculate gradients for G_A
        self.optimizer_G_A.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D_A.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.optimizer_D_A.step()  # update D_A and D_B's weights
        
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        slice_name = self.slice_name
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
  
        return visual_ret, slice_name
    
    def get_label_tensor(self, img_tensor, params = [1, 1, 1, 1.85, 0.5]):
        
        img_tensor = (1 - img_tensor) / 2
        a, b, c, positive_threshold, background_threshold = params
        avg_pool_8 = F.avg_pool2d(img_tensor, kernel_size=(8, 8))
        max_pool_256 = F.max_pool2d(avg_pool_8, kernel_size=(32, 32))
        max_pool_512 = F.max_pool2d(avg_pool_8, kernel_size=(64, 64))
        sum = max_pool_512.sum()
        if sum > positive_threshold:
            predict = 0
        elif sum <= positive_threshold and sum >= background_threshold:
            predict = 1
        else:
            predict = 2
        
        summed_tensor = torch.sum(max_pool_256, dim=0)
        predict_tensor = torch.where(summed_tensor > positive_threshold, 0,
                                    torch.where((summed_tensor <= positive_threshold) & (summed_tensor >= background_threshold), 1, 2))
        
        return predict, predict_tensor