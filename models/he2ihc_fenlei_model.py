import torch
import itertools
import numpy as np
from PIL import Image
from collections import OrderedDict
from util.image_pool import ImagePool
import torchvision.transforms as transforms
from .base_model import BaseModel
from . import networks
from . import consistency_model
import torchvision.models as models
import lpips
import os

#试试不加cyc，同时D不concate，然后不用L1，看看能不能吻合 发现基本齐
class HE2IHCfenleiModel(BaseModel):
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
            parser.add_argument('--lambda_identity', type=float, default=0.1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_lpips', type=float, default=10.0, help='weight for VGGlpips loss')
            parser.add_argument('--lambda_pred', type=float, default=10.0, help='weight for prediction loss')
            parser.add_argument('--lambda_consist', type=float, default=0.0, help='weight for consistency loss')
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
        self.loss_names = ['D_A', 'G_A', 'G_GAN', 'G_lpips_A', 'G_L1'] #加了个G_consist
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
        self.netG_A = networks.define_G(opt.input_nc+1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        #从代码的一致性角度来看，该把它挪到networks里去, 但现在没啥用了
        """
        self.ConsistModel = consistency_model.NewResNet34().requires_grad_(False).eval()
        weight_path = '/home/a611/Projects/wu/he2ihc/consistency/checkpoint6/model_latest.pt'
        consist_state_dict = torch.load(weight_path, 'cpu')
        self.ConsistModel.load_state_dict(consist_state_dict)
        self.ConsistModel.to(self.device)
        """
        
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = networks.L1Loss(L=[20, 10, 10]).to(self.device)
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
        self.real_A = input['he'].to(self.device)
        self.real_B = input['ihc'].to(self.device)
        self.predict = input['predict'].to(self.device)

        predict_expanded = self.predict.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        predict_expanded = predict_expanded.expand(-1, 1, self.real_A.shape[2], self.real_A.shape[3])

        self.input = torch.cat((self.real_A, predict_expanded), dim=1)
        
        self.image_paths = input['he_path']
        self.slice_name = os.path.basename(self.image_paths[0]).split('_')[0] #这个slice_name是用来记录保存的训练图像来自哪个切片的, 因为一个batch里只保存第一张，所以只记录第一张的图像名

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.input)  # G_A(A)

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
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_G_A(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A

        # GAN loss D_A(G_A(A))
        self.loss_G_GAN = self.criterionGAN(self.netD_A(self.fake_B), True) * 2
        # 加个对应的lpips
        self.loss_G_lpips_A = self.criterionLpips(self.fake_B, self.real_B).mean() * self.opt.lambda_lpips
        # 把改动的L1全加上去
        self.loss_G_L1_64, self.loss_G_L1_32, self.loss_G_L1_16, self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_B)
        'G_L1_64', 'G_L1_32', 'G_L1_16', 'G_L1_8'
        # 加个对应的consistency loss
        """
        if not self.opt.lambda_consist == 0:
            labels = torch.zeros(self.real_A.shape[0]).to(self.device)
            img = self.process(self.real_A, self.fake_B, self.real_B)
            self.criterionConsist.eval()
            outputs = self.ConsistModel(img)
            self.loss_G_consist_A = self.criterionConsist(outputs, self.labels).sum() * self.opt.lambda_consist
            self.loss_G_A = self.loss_G_GAN + self.loss_G_lpips_A + self.loss_G_consist_A
        else:
            self.loss_G_A = self.loss_G_GAN + self.loss_G_lpips_A
        """
        self.loss_G_A = self.loss_G_GAN + self.loss_G_lpips_A + self.loss_G_L1
                        
        #self.loss_G_pred = self.criterionPred(self.real_B, self.fake_B).mean() * self.opt.lambda_pred
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
    
    def get_consistency(self):
        """
        with torch.no_grad():
            img = self.process(self.real_A, self.fake_B, self.real_B)
            outputs = self.ConsistModel(img)
            _, predicted = torch.max(outputs.data, 1) #0是right, 1是wrong
                        
            total_elements = predicted.numel()
            nonzero_elements = torch.count_nonzero(predicted)
            right_num = total_elements - nonzero_elements
        
        return right_num, total_elements
        """
        return 0, 0
    
    def process(self, real_A, fake_B, real_B):
            images = [real_A, fake_B, real_B]
            img = torch.cat(images, dim=1)
            return img