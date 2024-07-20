这个加到debug的launch.json里就能实现带命令行参数的debug
"args": ["--dataroot", "datasets/he2p63_512", "--name", "he2ihc_test_d8", "--model", "he2ihc", "--dataset_mode", "he2ihc", "--input_nc", "3", "--gpu_ids", "1", "--preprocess", "none"]

"args": ["--dataroot", "datasets/he2p63_p2p", "--name", "he2ihc_pix2pix", "--model", "pix2pix", "--gpu_ids", "1", "--preprocess", "none"]

## p2p resnet_9blocks baseline啥都不改 论L1 resnet好像还不如unet256, 虽然unet256是过拟合，但或许数据量上来之后会更好？
python train.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_resnet9b --model pix2pix --gpu_ids 0 --preprocess none --batch_size 4 --netG resnet_9blocks

python test.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_resnet9b --model pix2pix --gpu_ids 0 --preprocess none --batch_size 4 --netG resnet_9blocks

## p2p unet baseline啥都不改 d_loss一直在0.6左右震荡，gan_loss有下降趋势
python train.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_unet256 --model pix2pix --gpu_ids 0 --preprocess none --batch_size 4 --netG unet_256

python test.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_unet256 --model pix2pix --gpu_ids 0 --preprocess none --batch_size 4 --netG unet_256

python test.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_unet256 --model pix2pix --gpu_ids 1 --preprocess none --batch_size 4 --netG unet_256

## p2p unet baseline, netD改成pixel 结果糊的一比
python train.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_unet256_pixel --model pix2pix --gpu_ids 0 --preprocess none --batch_size 8 --netG unet_256 --netD pixel

python test.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_unet256_pixel --model pix2pix --gpu_ids 0 --preprocess none --batch_size 4 --netG unet_256 --netD pixel

## p2p resnet_9blocks baseline, netD改成pixel 没跑完，但已经知道结果了，必糊
python train.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_resnet_9blocks_pixel --model pix2pix --gpu_ids 1 --preprocess none --batch_size 3 --netG resnet_9blocks --netD pixel

python test.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_resnet_9blocks_pixel --model pix2pix --gpu_ids 1 --preprocess none --batch_size 3 --netG resnet_9blocks --netD pixel 

## p2p unet baseline, netD改成n_layers n_layers = 4 效果不好，看loss应该是鉴别器D压过了生成器G，实际上也确实如此
python train.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_unet256_nlayer4 --model pix2pix --gpu_ids 0 --preprocess none --batch_size 8 --netG unet_256 --netD n_layers --n_layers_D 4

python test.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_unet256_nlayer4 --model pix2pix --gpu_ids 0 --preprocess none --batch_size 8 --netG unet_256 --netD n_layers --n_layers_D 4

## p2p resnet_9blocks baseline, netD改成n_layers n_layers = 4 更是鉴别器拉满了,GAN_loss一路飙升，L1在8左右剧烈震荡，D_loss几乎降到0
python train.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_resnet_9blocks_nlayer4 --model pix2pix --gpu_ids 1 --preprocess none --batch_size 3 --netG resnet_9blocks --netD n_layers --n_layers_D 4

python test.py --dataroot ./datasets/he2p63_p2p --name he2p63_p2p_resnet_9blocks_nlayer4 --model pix2pix --gpu_ids 1 --preprocess none --batch_size 3 --netG resnet_9blocks --netD n_layers --n_layers_D 4

##  p2p unet baseline, 试试256的图 --model p2phe2ihc只是把setinput稍稍改动使得它能适配he2ihc dataset，别的不变 结果是比较糊 虽然unet一直都有点糊，但这个更糊了，也可能是还没完全收敛？亦或者256就是没办法？ 而分析loss后发现256的discriminator似乎都看不出来，但肉眼可见的糊
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_unet256_res256 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 48 --netG unet_256 --resize 256

python test.py --dataroot ./datasets/he2p63 --name he2p63_p2p_unet256_res256 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 48 --netG unet_256 --resize 256

##  p2p resnet baseline, 试试256的图 --model p2phe2ihc只是把setinput稍稍改动使得它能适配he2ihc dataset，别的不变 和unet一样比较糊，也是要么没收敛要么256就是如此
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_resnet9b_res256 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess none --batch_size 16 --netG resnet_9blocks --resize 256

python test.py --dataroot ./datasets/he2p63 --name he2p63_p2p_resnet9b_res256 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess none --batch_size 16 --netG resnet_9blocks --resize 256

##  p2p unet baseline, 试试1024的图 --model p2phe2ihc只是把setinput稍稍改动使得它能适配he2ihc dataset，别的不变 1024时unet对训练集的拟合效果非常好，但测试效果很拉，显然是过拟合了，但我觉得unet有潜力，而且resnet太慢了，或许之后转向unet 分析loss之后发现大概140个epoch之后D有压过G的趋势
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_unet256_res1024 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 8 --netG unet_256 --resize 1024

python test.py --dataroot ./datasets/he2p63 --name he2p63_p2p_unet256_res1024 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 8 --netG unet_256 --resize 1024

python test.py --dataroot ./datasets/HE2P63_3_shutiled --name he2p63_p2p_unet256_res1024 --model pix2pix  --gpu_ids 0 --preprocess none --batch_size 8 --netG unet_256 --num_test 1000 --results_dir results/he2p63_p2p_unet256_res1024/test_HE2P63_3_shutiled

## p2p resnet baseline, 试试1024的图 --model p2phe2ihc只是把setinput稍稍改动使得它能适配he2ihc dataset，别的不变
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_resnet9b_res1024 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess none --batch_size 1 --netG resnet_9blocks --resize 1024 

python test.py --dataroot ./datasets/he2p63 --name he2p63_p2p_resnet9b_res1024 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess none --batch_size 2 --netG resnet_9blocks --resize 1024

python test.py --dataroot ./datasets/HE2P63_3_shutiled --name he2p63_p2p_resnet9b_res1024 --model pix2pix  --gpu_ids 1 --preprocess none --batch_size 2 --netG resnet_9blocks --results_dir results/he2p63_p2p_resnet9b_res1024/test_HE2P63_3_shutiled

## 试试稍稍改了的he2ihc模型，把之前的改动全部用参数控制，这样整理起来方便
python train.py --dataroot ./datasets/he2p63 --name he2p63_he2ihc_unet256_res1024 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 1 --netG unet_256 --resize 1024 --use_seg

python test.py --dataroot ./datasets/he2p63 --name he2p63_he2ihc_unet256_res1024 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 4 --netG unet_256 --resize 1024

## 再试cyclegan，生成器用unet256，1024分辨率
python train.py --dataroot ./datasets/he2p63 --name he2p63_cyc_unet_res1024 --model cyche2ihc --dataset_mode he2ihc --preprocess none --batch_size 1 --norm batch --gpu_ids 1 --resize 1024 --netG unet_256

python test.py --dataroot ./datasets/he2p63 --name he2p63_cyc_unet_res1024 --model cyche2ihc --dataset_mode he2ihc --preprocess none --batch_size 1 --norm batch --gpu_ids 1 --resize 1024 --netG unet_256

## 稍稍改了的he2ihc模型，把生成器改成unet_512
python train.py --dataroot ./datasets/he2p63 --name he2p63_he2ihc_unet512_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 3 --netG unet_512 --resize 512 --use_seg --lambda_seg 0

python test.py --dataroot ./datasets/he2p63 --name he2p63_he2ihc_unet512_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 3 --netG unet_512 --resize 512 --use_seg

## 新的H2ER，先用p2p resnet_9blocks baseline和unet256 baseline
python train.py --dataroot ./datasets/H2ER_p2p --name H2ER_p2p_resnet9b --model pix2pix --gpu_ids 0 --preprocess none --batch_size 1 --netG resnet_9blocks

python test.py --dataroot ./datasets/H2ER_p2p --name H2ER_p2p_resnet9b --model pix2pix --gpu_ids 0 --preprocess none --batch_size 5 --netG unet_256

## 试试p2p做做分割？要么用p2p分割的与训练模型基础上训练？顺便把源码的预处理添加到he2ihc datasetmode了，所以之后的命令都有所改动
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2pseg_unet512_res512 --model p2pseg --dataset_mode he2ihc --gpu_ids 0 --preprocess resize_and_crop --batch_size 6 --netG unet_512 --load_size 584 --crop_size 512 --output_nc 3

python test.py --dataroot ./datasets/he2p63 --name he2p63_p2pseg_unet512_res512 --model p2pseg --dataset_mode he2ihc --gpu_ids 0 --preprocess resize --load_size 512 --batch_size 6 --netG unet_512 --output_nc 3

python train.py --dataroot ./datasets/he2p63 --name he2p63_p2pseg_unet512_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess resize_and_crop --batch_size 6 --netG unet_512 --load_size 584 --crop_size 512 --output_nc 3 --lambda_L1 10 --lambda_L1_d8 200 --continue_train

## 试试新的网络模型 从Unet开始 34M
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_Unet_res512 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess resize_and_crop --batch_size 5 --netG Unet --load_size 554 --crop_size 512 --output_nc 3 --gpu_ids 1

python test.py --dataroot ./datasets/he2p63 --name he2p63_p2p_Unet_res512 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess resize --load_size 512 --batch_size 5 --netG Unet  --output_nc 3 --gpu_ids 1

## 试试新的网络模型 AttUnet 效果都很屎
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_AttUnet_res512 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess resize_and_crop --batch_size 4 --netG AttUnet --load_size 554 --crop_size 512 --output_nc 3 --gpu_ids 0

python test.py --dataroot ./datasets/he2p63 --name he2p63_p2p_AttUnet_res512 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess resize --load_size 512 --batch_size 4 --netG AttUnet  --output_nc 3 --gpu_ids 0

## AttUnet 试试1024分辨率
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_AttUnet_res1024 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess resize --batch_size 1 --netG AttUnet --load_size 1024  --output_nc 3 --gpu_ids 1

python test.py --dataroot ./datasets/he2p63 --name he2p63_p2p_AttUnet_res1024 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess resize --load_size 1024 --batch_size 1 --netG AttUnet  --output_nc 3 --gpu_ids 1

## unet128 试试 512分辨率开始
python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_unet128_res512 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess resize --batch_size 16 --netG unet_128 --load_size 512  --output_nc 3 --gpu_ids 0

python train.py --dataroot ./datasets/he2p63 --name he2p63_p2p_unet128_res512 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess resize --batch_size 16 --netG unet_128 --load_size 512  --output_nc 3 --gpu_ids 0

## he2ihc
python train.py --dataroot ./datasets/he2p63 --name he2p63_he2ihc_unet256_res1024 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 1 --netG unet_256 --resize 1024 --use_seg

python test.py --dataroot ./datasets/he2p63 --name he2p63_he2ihc_unet256_res1024 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --preprocess none --batch_size 4 --netG unet_256 --resize 1024

## HE2H p2p resnet9b 512分辨率 baseline 直接mode collapse
python train.py --dataroot datasets/HE2H --name HE2H_p2p_resnet9b_res512 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess none --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512

python test.py --dataroot ./datasets/HE2H --name HE2H_p2p_resnet9b_res512 --model p2phe2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess none --batch_size 2 --netG resnet_9blocks --preprocess resize --load_size 512

 # HE2H he2ihc unet256 res512 试试能不能避免mode collapse 
python train.py --dataroot datasets/HE2H --name HE2H_he2ihc_unet256_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess none --batch_size 8 --netG unet_256 --preprocess resize --load_size 512

python train.py --dataroot datasets/HE2H --name HE2H_he2ihc_unet256_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 1 --preprocess none --batch_size 8 --netG unet_256 --preprocess resize --load_size 512

## 把源码的预处理加到he2ihc dataset_mode里了，并删去了he2ihc_model中使用的seg功能，感觉分割没用，下面使用he2ihc的模板示例，HE2ER的
python train.py --dataroot datasets/HE2ER --name HE2ER_he2ihc_resnet9b_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 

python test.py --dataroot datasets/HE2ER --name HE2ER_he2ihc_resnet9b_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 

# 试试unet
python train.py --dataroot datasets/HE2ER --name HE2ER_he2ihc_unet256_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG unet_256 --preprocess resize --load_size 512 

python test.py --dataroot datasets/HE2ER --name HE2ER_he2ihc_unet256_res512 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG unet_256 --preprocess resize --load_size 512 

## 尝试加入cycleloss 先试试he2p63
# resnet9b 还是resnet9b效果好的多
python train.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 

python test.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 50

# 试试不上lpips做对比
python train.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512_lpips0 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --lambda_lpips 0

python test.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512_lpips0 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 50

# lpips权重调高做对比 100
python train.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512_lpips100 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --lambda_lpips 100

python test.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512_lpips100 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 50

# lpips权重调高做对比 40
python train.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512_lpips40 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --lambda_lpips 40

python test.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512_lpips40 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 50

# lpips权重调高做对比 20 同时训练epoch降为50 + 100, 初始学习率调至0.0003
python train.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512_lpips20 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --lambda_lpips 20

python test.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_resnet9b_res512_lpips20 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 50

## lpips 20, epoch 70+100, 学习率0.0002
python train.py --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips20 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --lambda_lpips 20

python test.py --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips20 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 200

## lpips 15, epoch 70+100, 学习率0.0002 试试新数据
python train.py --dataroot datasets/HE2CD138 --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips15 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --lambda_lpips 15

python test.py --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips15 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 200

## lpips 15, epoch 70+100, 学习率0.0002 试试新数据 5000张试试unet
python train.py --dataroot datasets/HE2CD138 --name HE2ER_ttcyche2ihc_unet_res512_lpips15 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG unet_256 --preprocess resize --load_size 512 --lambda_lpips 15

python test.py --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips20 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG unet_256 --preprocess resize --load_size 512 --num_test 200


# unet
python train.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_unet256_res512 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG unet_256 --preprocess resize --load_size 512 

python test.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_unet256_res512 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG unet_256 --preprocess resize --load_size 512 --num_test 50
# att unet
python train.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_AttUnet_res512 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG unet_256 --preprocess resize --load_size 512 --lambda_lpips 10

python test.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_AttUnet_res512 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 3 --netG AttUnet --preprocess resize --load_size 512 --num_test 50

# R2attunet 奇怪 无论batchsize多少都一样爆显存，感觉是模型有问题
python train.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_AttUnet_res512 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 2 --netG R2AttUnet --preprocess resize --load_size 512 --lambda_lpips 10

python test.py --dataroot datasets/he2p63 --name he2p63_ttcyche2ihc_AttUnet_res512 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 3 --netG R2AttUnet --preprocess resize --load_size 512 --num_test 50



## 测试p2p模型能不能收敛 它可以，那就显得非常奇怪
python train.py --dataroot ./datasets/he2p63_p2p --name he2p63_pix2pix --model pix2pix --direction AtoB --netG resnet_9blocks --preprocess none

python train.py --dataroot ./datasets/he2p63_p2p --name he2p63_pix2pix --model pix2pix --direction AtoB --netG resnet_9blocks --preprocess none

## 试试p2p模型用he2ihc数据集能不能收敛 它也能 那tm就起了他妈的怪了
python train.py --dataroot ./datasets/he2p63_512 --name he2p63_pix2pix --model pix2pix --dataset_mode he2ihc --direction AtoB --netG resnet_9blocks --preprocess none

## 试试上面的p2p模型只改个名能不能收敛 可以 把default改改，少输两个参数 还是能收敛 batch_size 6刚好把显存拉满 到快40个epoh了仍然非常糊，我不理解
python train.py --dataroot ./datasets/he2p63_512 --name he2p63_jihubudong --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 6 --gpu_ids 1 

## 加上了降采样L1 loss woc 至少Dloss没有直接降到0, md 这让我更困惑了
python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8 --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 6 --gpu_ids 1
python test.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8 --model he2ihc --dataset_mode he2ihc --preprocess none --gpu_ids 1 

## 去了d8 只用lpips
python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_VGG --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 4 --gpu_ids 1
python test.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8 --model he2ihc --dataset_mode he2ihc --preprocess none --gpu_ids 1


## 但上面这些除了p2p好像都收敛的非常慢，亦或者根本收敛不了？ 太令人疑惑了，即使想做实验卡也占满了，目前觉得最有可能的就是Adam改为AdamW造成的，可以试试在这基础上调调学习率，看了，能收敛但是比较慢，d8好像看不出明显差距？

## 试试 这个还没试
python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8 --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 4 --gpu_ids 1 --lr 0.0001 --n_epochs 50 --n_epochs_decay 100

## 再上个感知模型loss试试
python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8andVGG --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 4 --norm instance --gpu_ids 1 

python test.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8andVGG --model he2ihc --dataset_mode he2ihc --preprocess none --norm instance --gpu_ids 1 


## 只把输入改成heseg但不加分割loss
python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8VGGheseg --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 3 --norm instance --gpu_ids 0 --input_nc 4 --output_nc 3

python test.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8VGGheseg --model he2ihc --dataset_mode he2ihc --preprocess none --norm instance --gpu_ids 0 --input_nc 4 --output_nc 3

## 调试分割模型用的
python temp.py --dataroot ./datasets/he2p63_512

## 先让加了分割模型的能跑通 跑通了，但感觉不出效果，不过说实话没退步已经不错了
python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8VGGhesegloss --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 2 --norm instance --gpu_ids 1 --input_nc 4 --output_nc 3 

python test.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8VGGhesegloss --model he2ihc --dataset_mode he2ihc --preprocess none --norm instance --gpu_ids 1 --input_nc 4 --output_nc 3
## 改改参数再训一个试试
python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8VGGhesegloss1 --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 3 --norm instance --gpu_ids 0 --input_nc 4 --output_nc 3 --gan_mode lsgan --netD n_layers --n_layers_D 4 --lr 0.0001


python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_d8VGGhesegloss --model he2ihc --dataset_mode he2ihc --preprocess none --batch_size 4 --norm batch --gpu_ids 1 --input_nc 4 --output_nc 3 --

## 我觉得没必要把cyclegan加进去训练，直接用训好的更好
python train.py --dataroot ./datasets/he2p63_512 --name he2ihc_cyclegan --model cycle_gan --dataset_mode he2ihc --preprocess none --batch_size 1 --norm instance --gpu_ids 0 --lambda_identity 0

## 试试拿p2p做分割？







python train.py --dataroot /home/k611/data/yi/pytorch-CycleGAN-and-pix2pix-master_att/datasets/Dataset_daoru/P63 --name HE2P63_ttcyche2ihc_unet_512_res512_lpips15 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG unet_512 --preprocess resize --load_size 512 --lambda_lpips 15

python test.py --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips15 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 200

## 训55个epoch的加了预测网络的，md 好像是反向的效果....
python train.py --dataroot datasets/HE2ER --name HE2ER_tttcyche2ihc_resnet9b_res512_lpips20_pred10 --model tttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --lambda_lpips 20 --lambda_pred 10

python test.py --dataroot datasets/HE2ER --name HE2ER_tttcyche2ihc_resnet9b_res512_lpips20_pred10 --model tttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 200

## 训50个epoch的加了预测网络的，lambda_pred 改到50
python train.py --dataroot datasets/HE2ER --name HE2ER_tttcyche2ihc_resnet9b_res512_lpips20_pred50 --model tttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --lambda_lpips 20 --lambda_pred 50

python test.py --dataroot datasets/HE2ER --name HE2ER_tttcyche2ihc_resnet9b_res512_lpips20_pred50 --model tttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 200

## 拿个不加的50个epoch的对比一下
python train.py --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips20 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --epoch 50

python test.py --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips20 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --epoch 50 --num_test 200

## 改了一下读取参数的方法 试试能不能用
python train.py --train_root /home/k611/data2/wu/he2ihc/datasets/HE2ER/train --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips20_temp --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --epoch 50

python test.py --test_root /home/k611/data2/wu/he2ihc/datasets/HE2ER/test_temp --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips20 --model ttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --epoch 50 --num_test 200

## 加入wandb，测试用 
## 发现用wandb不太行，网络老有问题，虽说是可以在本地.... 要不试试在本地
python train_beta.py --dataroot /home/k611/data2/Dataset/P63_ruxian --name betatest --model tttcyche2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --display_freq 100

python test.py --test_root /home/k611/data2/wu/he2ihc/datasets/HE2ER/test_temp --dataroot datasets/HE2ER --name HE2ER_ttcyche2ihc_resnet9b_res512_lpips20 --model tttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --epoch 50 --num_test 200

## 改动过的 改成D输入concate的了，但怎么说呢...可能会影响到吻合度 似乎确实会影响
python train_beta.py --dataroot /home/k611/data2/Dataset/P63_ruxian --name betatest3 --model tttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --display_freq 100

## betatest4 D改回不concate的了，然后给ihc到he加了L1和lpips，希望能让它正常点
python train_beta.py --dataroot /home/k611/data2/Dataset/P63_ruxian --name betatest4 --model tttcyche2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --display_freq 100

## 如果我不cyc, 然后D改成不concate，然后不用L1 loss 是不是也能吻合？我觉得有戏
python train_beta.py --dataroot /home/k611/data2/Dataset/P63_ruxian --name betatest5 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 3 --netG resnet_9blocks --preprocess resize --load_size 512 --display_freq 100

python test.py --dataroot /home/k611/data2/Dataset/P63_ruxian --test_root /home/k611/data2/Dataset/P63_ruxian/test_A007418 --name betatest5 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 3 --netG resnet_9blocks --preprocess resize --load_size 512

python test.py --dataroot /home/k611/data2/Dataset/P63_ruxian --test_root /home/k611/data2/Dataset/P63_ruxian/test_A012607_3_ruai_Y --name betatest5 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 3 --netG resnet_9blocks --preprocess resize --load_size 512

## woc 不cyc好像真的行，现在把2048*2048的拿来试试，先还是降采样两倍 betatest7是换了很全的数据集的
python train_beta.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_2048 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 2 --netG resnet_9blocks --preprocess resize --load_size 1024 --display_freq 100

python test.py --dataroot /home/k611/data2/Dataset/P63_ruxian --test_root /home/k611/data3/wu/Dataset/P63_ruxian/A007418_test_2048 --name betatest6 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 2 --netG resnet_9blocks --preprocess resize --load_size 1024 --num_test 5000 --epoch 50

python test.py --dataroot /home/k611/data2/Dataset/P63_ruxian --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_2048/test_A012607 --name betatest6 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 2 --netG resnet_9blocks --preprocess resize --load_size 1024 --num_test 5000 --epoch 50

## 跑一下新数据集的p63
python /home/s611/Projects/wu/he2ihc/train_beta.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --display_freq 100 --continue_train --epoch_count 50

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A012607 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/A012607

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/test_C113327 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./result_temp/

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/test_A007418 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./result_A007418/

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/test_A15520 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./result_A15520/

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/train --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./result_train/

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A16746 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./A16746

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A16886 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./A16886

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A17244 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./A17244

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/C152221 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./C152221

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A007418 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./A007418

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A009798 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/A009798

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A013564 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/A013564 

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A13923 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/A13923

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A14053 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/A14053

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/C104494 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/C104494

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A15331 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/A15331

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A14946 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/A14946

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/A14946 --name betatest7 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./results/A14946


## 跑一下很随意的新模型 感觉这种随便改改的模型没法用
python train_beta.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --name betatest8 --model ttthe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 1 --netG ResnetAtt6blocks --preprocess resize --load_size 512 --display_freq 100

## 跑一下新数据集的p63的消融测试吧
python train_beta.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --name betatest9 --model pix2pixhe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --display_freq 100 --continue_train

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/test_A012607 --name betatest9 --model pix2pixhe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./result_A012607/

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/test_C113327 --name betatest9 --model pix2pixhe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./result_C113327/

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/test_A007418 --name betatest9 --model pix2pixhe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./result_A007418/

python test.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/test_A15520 --name betatest9 --model pix2pixhe2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --num_test 100000 --results_dir ./result_A15520/



## 跑一下he2he转完的训练
python train_beta.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --name he2hexunlian --model ttthe2ihc --dataset_mode he2he --gpu_ids 1 --batch_size 32 --netG resnet_9blocks --preprocess resize --load_size 256 --display_freq 100 --crop_size 256 --n_epochs 50 --n_epochs_decay 50

## 加入consistency_model测试，先debug
python train_beta2.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --name betatest_consist_debug --model tttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --display_freq 100

python test_beta.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/C113327 --name betatest7 --model tttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --results_dir ./results/temp --num_test 10000

python test_beta.py --dataroot /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --test_root /home/s611/Projects/wu/Dataset/P63_ruxian_1024/temp_for_consistency --name betatest7 --model tttthe2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --results_dir ./results/temp_for_consistency --num_test 1000

## ECAD训练 先在171上补一个256的
python train_beta2.py --name betatest_ECAD --model tttthe2ihc --dataset_mode he2ihctest --gpu_ids 1 --batch_size 15 --netG resnet_9blocks --preprocess resize --load_size 256 --crop_size 256 --src_path /home/s611/Projects/wu/Dataset/ECAD_ruxian_1024 --n_epochs 100 --n_epochs_decay 100 --slice_list A8827 C133447 A15331 A15520 C152072

python test_beta2.py --name betatest_ECAD_256 --model tttthe2ihc --dataset_mode he2ihctest --gpu_ids 1 --batch_size 15 --netG resnet_9blocks --preprocess resize --load_size 256 --crop_size 256 --src_path /home/s611/Projects/wu/Dataset/ECAD_ruxian_1024 --slice_list A10032 A16886 A17244 C152221 C136881 --num_test 1000000

## P120训练 
python train_beta2.py --name betatest_P120 --model tttthe2ihc --dataset_mode he2ihctest --gpu_ids 0 --batch_size 17 --netG resnet_9blocks --preprocess resize --load_size 256 --crop_size 256 --src_path /home/s611/Projects/wu/Dataset/P120_ruxian_1024 --n_epochs 100 --n_epochs_decay 100 --slice_list A8827 C133447 A15331 A15520 C152072 

## p16
python train_beta2.py --name betatest_P16 --model tttthe2ihc --dataset_mode he2ihctest --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/P16_gongjin_1024 --n_epochs 100 --n_epochs_decay 100 --slice_list A03308 A011751 A012617

python test_beta2.py --name betatest_P16 --model tttthe2ihc --dataset_mode he2ihctest --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --crop_size 512 --src_path /home/s611/Projects/wu/Dataset/P16_gongjin_1024 --slice_list A03308 A011751 A012617 A012818 --num_test 1000000

python test_beta2.py --name betatest_P16 --model tttthe2ihc --dataset_mode he2hetest --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --crop_size 512 --src_path /home/s611/Projects/wu/Dataset/cin --slice_list cin1-A015404 cin2-A04272 cin3_A014080 yan_A006484 --num_test 1000000 --epoch 200 --results_dir ./results/cin

## 训个CK56
python train_beta2.py --name betatest_CK56 --model tttthe2ihc --dataset_mode he2ihctt --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/CK56_ruxian_1024 --n_epochs 70 --n_epochs_decay 80 --slice_list A007418 A14946 A15331 A15520 A16476 A16886 A17244 C105272 C105560 C152221 N05A14053 --continue_train --epoch_count 90

python train_beta2.py --name betatest_CK56_256 --model tttthe2ihc --dataset_mode he2ihctt --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 256 --crop_size 256 --src_path /home/s611/Projects/wu/Dataset/CK56_ruxian_1024 --n_epochs 70 --n_epochs_decay 80 --slice_list A007418 A14946 A15331 A15520 A16476 A16886 A17244 C105272 C105560 C152221 N05A14053 --continue_train --epoch_count 30

python test_beta2.py --name betatest_CK56 --model tttthe2ihc --dataset_mode he2ihctt --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/CK56_ruxian_1024 --slice_list N5A13923 C141449 C151314 --num_test 10000000

python test_beta2.py --name betatest_CK56_256 --model tttthe2ihc --dataset_mode he2ihctt --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/CK56_ruxian_1024 --slice_list N5A13923 C141449 C151314 --num_test 10000000



/home/s611/Projects/Frozen_Slides_tianjin/Data_Crop/keshan/results/


## 256的全p63 150epoch训练 betatest_all 测试一下刚给的/home/s611/Projects/Frozen_Slides_tianjin/Data_Crop/keshan/results/JiangxiFrozen_to_Jiangxi/test_latest/images/fake_B
python test.py --name betatest_all --model tttthe2ihc --dataset_mode he2he --gpu_ids 1 --batch_size 15 --netG resnet_9blocks --preprocess resize --load_size 256 --crop_size 256 --test_root /home/s611/Projects/Frozen_Slides_tianjin/Data_Crop/keshan/results/JiangxiFrozen_to_Jiangxi/test_latest/images/fake_B --num_test 1000000 --results_dir ./results/JiangXi_Frozen

## 江西 cin 训练
python train_beta2.py --name betatest_CK56 --model tttthe2ihc --dataset_mode he2ihctt --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/CK56_ruxian_1024 --n_epochs 70 --n_epochs_decay 80 --slice_list A007418 A14946 A15331 A15520 A16476 A16886 A17244 C105272 C105560 C152221 N05A14053 --continue_train --epoch_count 90

python train_beta2.py --name CIN_P16 --model tttthe2ihc --dataset_mode he2ihctest --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/CIN/P16 --n_epochs 100 --n_epochs_decay 100 --slice_list CA_A13034 CA_A15119 CACIN3_A03859 yan_A006484

python train_beta2.py --name CIN_Ki67 --model tttthe2ihc --dataset_mode he2ihctest --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/CIN/Ki67 --n_epochs 100 --n_epochs_decay 100 --slice_list CA_A13034 CA_A15119 CACIN3_A03859 yan_A006484

## CK56测试
python test_beta2.py --name betatest_CK56_256 --model tttthe2ihc --dataset_mode he2het --gpu_ids 1 --batch_size 15 --netG resnet_9blocks --preprocess resize --load_size 256 --crop_size 256 --test_root /home/s611/Projects/Frozen_Slides_tianjin/Data_Crop/keshan/results/JiangxiFrozen_to_Jiangxi/test_latest/images/fake_B --num_test 1000000 --results_dir ./results/JiangXi_Frozen

python test_beta2.py --name betatest_CK56_256 --model tttthe2ihc --dataset_mode he2het --gpu_ids 1 --batch_size 15 --netG resnet_9blocks --preprocess resize --load_size 256 --crop_size 256 --test_root /home/s611/Projects/lan/data/tianjinzhongliu_test/DCLGAN_atten_161to171_new_0611_epoch64/results/DCL_frozenFFPE20000/test_latest/images/fake_B --num_test 1000000 --results_dir ./results/Tianjinzhongliu

## P63测试
python test_beta.py --name betatest_all --model tttthe2ihc --dataset_mode he2he --gpu_ids 0 --batch_size 15 --netG resnet_9blocks --preprocess resize --load_size 256 --crop_size 256 --test_root /home/s611/Projects/lan/data/tianjinzhongliu_test/DCLGAN_atten_161to171_new_0611_epoch64/results/DCL_frozenFFPE20000/test_latest/images/fake_B --num_test 1000000 --results_dir ./results/Tianjinzhongliu


/home/s611/Projects/lan/data/tianjinzhongliu_test/DCLGAN_atten_161to171_new_0611_epoch64/results/DCL_frozenFFPE20000/test_latest/images/fake_B

## 加入分类条件的P63测试
python train_beta3.py --name betatest_P63fenlei --model ttttthe2ihc --dataset_mode he2ihctttt --gpu_ids 1 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /root/projects/wu/Dataset/P63_ruxian_1024 --n_epochs 70 --n_epochs_decay 80 --predict_path /root/projects/wu/classify_project/probs_save/IHC_probs_2/csv/probs.csv --slice_list A007418 A012607 A013564 A13923 A14053 A15331 A15520 A16746 A16886 A154421 C104494 C113327 

python test_beta3.py --name betatest_P63fenlei --model ttttthe2ihc --dataset_mode he2ihctttt --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /root/projects/wu/Dataset/P63_ruxian_1024 --predict_path /root/projects/wu/classify_project/probs_save/IHC_probs_2/csv/probs.csv --slice_list A009798 A14946 A17244 C105560 C152221 C152280 C152536 --num_test 10000000 --results_dir ./results --epoch 80

python test_beta3.py --name betatest_P63fenlei --model ttttthe2ihc --dataset_mode he2ihctttt --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /root/projects/wu/Dataset/test_P63_FROZEN --predict_path /root/projects/wu/classify_project/probs_save/HE_probs_2_FROZEN_test/csv/probs.csv  --slice_list B008490_frozen B008330_frozen B008243_frozen B008012_frozen B007928_frozen  --num_test 10000000 --results_dir ./results/test_P63_FROZEN --epoch 80

## consist model训练
python train.py --name betatest_consist --model consistcycle --dataset_mode consist --gpu_ids 0 --batch_size 1 --netG resnet_6blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --continue_train --epoch_count 30 --n_epochs 40 --n_epochs_decay 30

python test.py --name betatest_consist --model consistcycle --dataset_mode consist --gpu_ids 1 --batch_size 1 --netG resnet_6blocks --preprocess resize --load_size 512 --src_path /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --epoch latest --num_test 100000000




## 修改的代码，第一次运行debug
python train.py --name betatest_debug --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 4 --netG resnet_9blocks --preprocess resize --load_size 512 --src_path /home/f611/Projects/data/Dataset_171/P63_ruxian_1024 --n_epochs 70 --n_epochs_decay 80 --slice_list all --use_label

## 修改后的代码，用来拯救gongjin, 我觉得有机会
python train.py --name betatest_P16 --model he2ihc --dataset_mode he2ihc --gpu_ids 0 --batch_size 8 --netG resnet_9blocks --preprocess crop --load_size 1024 --crop_size 512 --src_path /home/f611/Projects/data/Dataset_171/CIN_new/P16 --n_epochs 80 --n_epochs_decay 80 --slice_list all --continue_train --epoch_count 67

## 跑个cyclegan的对比实验 P63的
python train.py --name betatest_P63_cyclegan --model cycle_he2ihc --dataset_mode he2ihc --gpu_ids 1 --batch_size 1 --netG resnet_9blocks --preprocess resize --load_size 512 --crop_size 512 --src_path /home/s611/Projects/wu/Dataset/P63_ruxian_1024 --n_epochs 80 --n_epochs_decay 80 --slice_list A007418 A013564 A13923 A14053 A15331 A16746 A16886 A154421 C104494 C113327 --use_train_list


