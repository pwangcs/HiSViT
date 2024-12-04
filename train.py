import torch
import torch.nn as nn
import time
import datetime
import einops
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import cv2
import scipy.io as scio
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from opts import parse_args
from test import test
from model.arch import HiSViT
from utils import load_checkpoint, checkpoint, TrainData, Logger, time2file_name


def train(args, network, optimizer, pretrain_epoch, logger, weight_path, result_path1, result_path2=None):
    criterion  = nn.MSELoss()
    criterion = criterion.to(args.device)
    rank = 0
    if args.distributed:
        rank = dist.get_rank() 
    dataset = TrainData(args)
    dist_sampler = None

    if args.distributed:
        dist_sampler = DistributedSampler(dataset, shuffle=True)
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            sampler=dist_sampler, num_workers=args.num_workers)
    else:
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for epoch in range(pretrain_epoch + 1, pretrain_epoch + args.epochs + 1):
        epoch_loss = 0
        network = network.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            if args.color_channels == 1:
                gt = data
                x = gt
            elif args.color_channels == 3:
                gt, x = data
            else:
                ValueError('there is an error in the input color channel.')

            gt = gt.float().to(args.device) # (b, 1, t, h w) or (b, 3, t, h w)
            x = x.float().to(args.device) # (b, 1, t, h w)
            args.mask = args.mask.float().to(args.device)
            args.mask_s = args.mask_s.float().to(args.device)

            # imaging model 
            meas= torch.mul(args.mask, x).sum(dim=2, keepdim=True) # (b 1 1 h w)

            # initialization: 2D -> 3D
            meas_re = torch.div(meas, args.mask_s)  # (b 1 1 h w)
            # print(gt.shape,x.shape,meas.shape,meas_re.shape)
            x = meas_re + torch.mul(args.mask, meas_re)  # (b 1 t h w)

            x = torch.cat((x, meas_re.repeat(1,1,x.shape[2],1,1)), dim=1)

            optimizer.zero_grad()
            out = network(x)

            loss = torch.sqrt(criterion(out, gt))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if rank==0 and (iteration % args.iter_step) == 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info('epoch: {:<3d}, iter: {:<4d}, loss: {:.5f}, lr: {:.6f}.'
                            .format(epoch, iteration, loss.item(), lr))

            if rank==0 and (iteration % args.save_train_image_step) == 0:
                img_out = out[0].detach()  # 1 t h w or 3 t h w
                img_gt = gt[0].detach()  # 1 t h w or 3 t h w
                img_out = einops.rearrange(img_out, 'c t h w -> h (t w) c')
                img_gt = einops.rearrange(img_gt, 'c t h w -> h (t w) c')
                img = torch.cat((img_gt, img_out), dim=0) *255                
                if args.color_channels == 1:
                    img = img.squeeze().cpu().numpy()
                elif args.color_channels == 3:
                    img = img.cpu().numpy()[:,:,::-1]
                img_path = './'+ result_path1+ '/'+'epoch_{}_iter_{}.png'.format(epoch, iteration)
                img = img.astype(np.float32)
                cv2.imwrite(img_path,img)

        end_time = time.time()
        if rank==0:
            logger.info('epoch: {}, avg. loss: {:.5f}, lr: {:.6f}, time: {:.2f}s.\n'.format(epoch, epoch_loss/(iteration+1), lr, end_time-start_time))

        if rank==0 and (epoch % args.save_model_step) == 0:
            model_out_path = './' + weight_path + '/' + 'epoch_{}.pth'.format(epoch)
            if args.distributed:
                checkpoint(epoch, network.module, optimizer, model_out_path)
            else:
                checkpoint(epoch, network, optimizer, model_out_path)
               
        if rank==0 and args.test_flag:
            logger.info('epoch: {}, psnr and ssim test results:'.format(epoch))
            if args.distributed:
                psnr_dict, ssim_dict = test(args, network.module, logger, result_path2)
            else:
                psnr_dict, ssim_dict = test(args, network, logger, result_path2)
            logger.info('psnr_dict: {}.'.format(psnr_dict))
            logger.info('ssim_dict: {}.\n'.format(ssim_dict))


if __name__ == '__main__':
    args = parse_args()
    rank = 0
    pretrain_epoch = 0
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    if rank ==0:
        result_path1 = 'results'  + '/' + date_time + '/train'
        weight_path = 'weights'  + '/' + date_time
        log_path = 'log/log' 
        if not os.path.exists(result_path1):
            os.makedirs(result_path1,exist_ok=True)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path,exist_ok=True)
        if not os.path.exists(log_path):
            os.makedirs(log_path,exist_ok=True)
        if args.test_flag:
            result_path2 = 'results' + '/' + date_time + '/test'
            if not os.path.exists(result_path2):
                os.makedirs(result_path2,exist_ok=True)
        else:
            result_path2 = None

        if args.color_channels == 1:
            mask_path = './mask/gray_mask.mat'
        elif args.color_channels == 3:
            mask_path = './mask/color_mask.mat'
        mask = scio.loadmat(mask_path)['mask']
        mask = mask.transpose([2, 0, 1]).astype(np.float32)
        mask = mask[:args.B,:args.size[0],:args.size[1]]
        mask_s = np.sum(mask, axis=0)
        mask_s[mask_s == 0] = 1
        args.mask = torch.from_numpy(mask)
        args.mask_s = torch.from_numpy(mask_s)
        args.mask = einops.rearrange(args.mask,'t h w->1 1 t h w')
        args.mask_s = einops.rearrange(args.mask_s,'h w->1 1 1 h w')
        
    logger = Logger(log_path)
    
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device('cuda',local_rank)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    network = HiSViT(dim=args.dim, frames=args.B, size=args.size, color_ch=args.color_channels, blocks=args.blocks).to(args.device)       
    optimizer = optim.Adam([{'params': network.parameters()}], lr=args.lr)
    
    if rank==0:
        if args.pretrained_model_path is not None:
            logger.info('Loading pretrained model...')
            pretrained_dict = torch.load(args.pretrained_model_path)
            if 'pretrain_epoch' in pretrained_dict.keys():
                pretrain_epoch = pretrained_dict['pretrain_epoch']
                logger.info('Pretrain epoch: {}'.format(pretrain_epoch))
            load_checkpoint(network, pretrained_dict, logger)
        else:
            logger.info('No pretrained model.')

    if args.distributed:
        network = DDP(network, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train(args, network, optimizer, pretrain_epoch, logger, weight_path, result_path1, result_path2)


