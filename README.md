# [[ECCV 2024] Hierarchical Separable Video Transformer for Snapshot Compressive Imaging](https://arxiv.org/abs/2407.11946)

[Ping Wang](https://scholar.google.com/citations?user=WCsIUToAAAAJ&hl), [Yulun Zhang](https://scholar.google.com/citations?user=ORmLjWoAAAAJ&hl), [Lishun Wang](https://scholar.google.com/citations?user=BzkbrCgAAAAJ&hl), [Xin Yuan](https://scholar.google.com/citations?user=cS9CbWkAAAAJ&hl)

#### Video SCI Reconstruction Task and Its Degradation Analysis
![pipeline](https://github.com/user-attachments/assets/783424c1-0a1e-4291-b5ed-be37fd5b5ac8)

#### Video SCI Reconstruction Architecture
![arch](https://github.com/user-attachments/assets/cff201f2-d1f9-4e4c-8db4-9fe206b43c06)
#### HiSViT (building block)
![hisvit](https://github.com/user-attachments/assets/916cb9cb-acda-4a20-8258-b42173e855e5)
#### Result
![sumary](https://github.com/user-attachments/assets/1518e543-83bd-4621-9442-569cc3419ae6)


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


## Citation
If you use HiSViT, please consider citing:
```
@inproceedings{wang2025hierarchical,
  title={Hierarchical Separable Video Transformer for Snapshot Compressive Imaging},
  author={Wang, Ping and Zhang, Yulun and Wang, Lishun and Yuan, Xin},
  booktitle={European Conference on Computer Vision},
  pages={104--122},
  year={2025},
  organization={Springer}
}
```




