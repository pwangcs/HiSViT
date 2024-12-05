# [[ECCV 2024] Hierarchical Separable Video Transformer for Snapshot Compressive Imaging](https://arxiv.org/abs/2407.11946)

[Ping Wang](https://scholar.google.com/citations?user=WCsIUToAAAAJ&hl), [Yulun Zhang](https://scholar.google.com/citations?user=ORmLjWoAAAAJ&hl), [Lishun Wang](https://scholar.google.com/citations?user=BzkbrCgAAAAJ&hl), [Xin Yuan](https://scholar.google.com/citations?user=cS9CbWkAAAAJ&hl)

## Preparation

#### data

Please download **testdata** from https://drive.google.com/drive/u/0/folders/175mGay-6FS_B4dDNTUkayzMbikHfnsit and download **traindata** from DAVIS2017 dataset (480p) https://davischallenge.org/davis2017/code.html

#### checkpoint

Pretrained models are available at https://drive.google.com/drive/u/0/folders/175mGay-6FS_B4dDNTUkayzMbikHfnsit

#### mask

Masks are available at https://drive.google.com/drive/u/0/folders/175mGay-6FS_B4dDNTUkayzMbikHfnsit



## Testing

#### HiSViT-9 Testing on gray dataset

python test.py --color_channels 1 --blocks 9 --size '[256,256]' --test_weight_path './checkpoint/hisvit9_gray.pth' --test_data_path './data/testdata/gray_256' 

#### HiSViT-13 Testing on gray dataset

python test.py --color_channels 1 --blocks 13 --size '[256,256]' --test_weight_path './checkpoint/hisvit13_gray.pth' --test_data_path './data/testdata/gray_256' 

#### HiSViT-9 Testing on color dataset

python test.py --color_channels 3 --blocks 9 --size '[512,512]' --test_weight_path './checkpoint/hisvit9_color.pth' --test_data_path './data/testdata/color_512' 



## Training


#### HiSViT-9 Training on gray dataset

###### Pretraining

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --distributed=True --color_channels 1 --blocks 9 --size '[128,128]' --epochs 100 --test_flag False --save_model_step 5

###### Fine-tuning

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --distributed=True --color_channels 1 --blocks 9 --size '[256,256]' --epochs 50 --test_flag True --save_model_step 1


#### HiSViT-13 Training on gray dataset

###### Pretraining

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --distributed=True --color_channels 1 --blocks 13 --size '[128,128]' --epochs 100 --test_flag False --save_model_step 5

###### Fine-tuning

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --distributed=True --color_channels 1 --blocks 13 --size '[256,256]' --epochs 50 --test_flag True --save_model_step 1


#### HiSViT-9 Training on color dataset

###### Pretraining

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --distributed=True --color_channels 3 --blocks 9 --size '[128,128]' --epochs 100 --test_flag False --save_model_step 5

###### Fine-tuning

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py --distributed=True --color_channels 3 --blocks 9 --size '[256,256]' --epochs 50 --test_flag True --save_model_step 1






