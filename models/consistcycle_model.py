import torch
import itertools
from collections import OrderedDict
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.models as models
import lpips
import os

#让转换出来的ihc颜色比较一致，先把训练集里的ihc都转到一组refer_ihc上，这是这个模型的作用
class consistcycleModel(BaseModel):
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
        self.loss_names = ['D_A', 'G_A', 'cycle_A_L1', 'cycle_A_lpips', 'idt_A', 'D_B', 'G_B', 'cycle_B_L1', 'cycle_B_lpips', 'idt_B'] #暂时把G_pred去掉了
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            self.netD_B_condition = networks.define_D(opt.input_nc + 1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            #self.fake_A_pool = ImagePool(opt.pool_size)  # 不知道这样有什么好处，但是要让predict对的上就没法用缓存了
            #self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionT = networks.TLoss().to(self.device)
            self.criterionCycle_L1 = torch.nn.L1Loss().to(self.device)
            self.criterionCycle_lpips = self.criterionLpips = lpips.LPIPS(net='vgg', eval_mode=True).eval().requires_grad_(False).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
  
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_A = torch.optim.AdamW(self.netG_A.parameters(), lr=opt.lr_G_A, betas=(opt.beta1, 0.999))
            self.optimizer_G_B = torch.optim.AdamW(self.netG_B.parameters(), lr=opt.lr_G_B, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.AdamW(self.netD_A.parameters(), lr=opt.lr_D_A, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.AdamW(self.netD_B.parameters(), lr=opt.lr_D_B, betas=(opt.beta1, 0.999))
            self.optimizer_D_B_condition = torch.optim.AdamW(self.netD_B_condition.parameters(), lr=opt.lr_D_B, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_B)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_B_condition)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['refer'].to(self.device) #这里AB反了，但也无所谓了
        self.real_B = input['ihc'].to(self.device)
        self.refer_path = input['refer_path']
        self.ihc_path = input['ihc_path']
        self.condition = input['predict']
        self.slice_name = input['slice'] #这个slice_name是用来记录保存的训练图像来自哪个切片的, 因为一个batch里只保存第一张，所以只记录第一张的图像名

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        self.real_A_conditioned = self.get_conditioned(self.real_A, self.condition)
        self.fake_A_conditioned = self.get_conditioned(self.fake_A, self.condition)

    def backward_D_basic(self, netD, real, fake, source):
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
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self): #改动过的
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B) 我不太理解这个的用处，但这里要让predict对的上不能这样做
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B, self.real_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        #fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A, self.real_B)
        self.loss_D_B_conditioned = self.backward_D_basic(self.netD_B_condition, self.real_A_conditioned, self.fake_A_conditioned, self.real_B)

    def backward_G_A(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A_L1 = self.criterionCycle_L1(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_A_lpips = self.criterionCycle_lpips(self.rec_A, self.real_A) * lambda_A
        #_, loss_T_discret = self.criterionT(self.real_A, self.fake_B, k = 8, real_params = [1.85, 0.5], fake_params = [1.85, 0.5]) # 现在就是要找到这个参数, 结果找不到, md
        # combined loss and calculate gradients
        
        self.loss_G_A = self.loss_G_A + self.loss_cycle_A_L1 + self.loss_cycle_A_lpips + self.loss_idt_A #如果是为了风格统一可以加一个ssim_loss之类的
        self.loss_G_A.backward()
        
    def backward_G_B(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_B * lambda_idt
        else:
            self.loss_idt_B = 0

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B_L1 = self.criterionCycle_L1(self.rec_B, self.real_B) * lambda_B
        self.loss_cycle_B_lpips = self.criterionCycle_lpips(self.rec_B, self.real_B) * lambda_B
        
        self.loss_G_B = self.loss_G_B + self.loss_cycle_B_L1 + self.loss_cycle_B_lpips + self.loss_idt_B
        self.loss_G_B.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G_A.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_G_B.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G_A()             # calculate gradients for G_A
        self.backward_G_B()             # calculate gradients for G_B
        self.optimizer_G_A.step()       # update G_A and G_B's weights
        self.optimizer_G_B.step() 
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D_A.zero_grad()   # set D_A and D_B's gradients to zero
        self.optimizer_D_B.zero_grad()
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D_A.step()  # update D_A and D_B's weights
        self.optimizer_D_B.step()
        
    def get_conditioned(self, image, condition):
        condition_tensor = condition.view(-1, 1, 1, 1).expand(-1, 1, image.shape[2], image.shape[3]).to(self.device)
        concated_tensor = torch.cat((image, condition_tensor), dim=1).to(self.device)
        
        return concated_tensor
         
        
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        slice_name = self.slice_name
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
  
        return visual_ret, slice_name
    
    def get_image_paths(self):
        return self.ihc_path
