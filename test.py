import torch 
import os
import cv2
import scipy.io as scio
import numpy as np 
import einops
from torch.utils.data import DataLoader 
from skimage.metrics import structural_similarity as ski_ssim
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from opts import parse_args
from model.arch import HiSViT
from utils import Logger, TestData, load_checkpoint

def test(args, network, logger, test_dir):
    network = network.eval()
    test_data = TestData(args) 
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)    

    if args.color_channels == 3:
        r = np.array([[1, 0], [0, 0]])
        g1 = np.array([[0, 1], [0, 0]])
        g2 = np.array([[0, 0], [1, 0]])
        b = np.array([[0, 0], [0, 1]])
        rgb2raw = np.zeros([3, args.size[0], args.size[1]])
        rgb2raw[0, :, :] = np.tile(r, (args.size[0] // 2, args.size[1] // 2))
        rgb2raw[1, :, :] = np.tile(g1, (args.size[0] // 2, args.size[1] // 2)) + np.tile(g2, (args.size[0] // 2, args.size[1] // 2))
        rgb2raw[2, :, :] = np.tile(b, (args.size[0] // 2, args.size[1] // 2))
        rgb2raw = einops.rearrange(rgb2raw, 'c h w-> 1 1 h w c')
        
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]

    for data in test_data_loader:
        gt = data
        if args.color_channels == 1:
            gt = gt[0].float().numpy()
            gt_rgb = gt
        elif args.color_channels == 3:
            gt, gt_rgb = gt
            gt = gt[0].float().numpy()
            gt_rgb = gt_rgb[0].float().numpy()
        
        b, t, h, w = gt.shape
        x = torch.from_numpy(gt).to(args.device)

        # imaging model 
        meas= torch.mul(args.mask, x)
        meas = torch.sum(meas, dim=1, keepdim=True)

        # initialization: 2D -> 3D
        meas_re = torch.div(meas, args.mask_s)
        x = meas_re + torch.mul(args.mask, meas_re)  # (b t h w)

        with torch.no_grad():
            inp = torch.cat((x.unsqueeze(1), meas_re.repeat(1,t,1,1).unsqueeze(1)), dim=1)
            if args.color_channels == 1:
                out = network(inp)
            else:
                out = torch.zeros((b,args.color_channels,t,h,w))
                for k in range(b): out[k:k+1] = network(inp[k:k+1])

        if args.color_channels == 1:
            out_rgb = out.clip(0,1).squeeze(1).cpu().numpy()
            out_pic = out_rgb
        else:
            out_rgb = out.clip(0,1).permute(0,2,3,4,1).cpu().numpy() # (b t h w c)
            out_pic = np.sum(rgb2raw*out_rgb, axis=-1)
        psnr_t = 0
        ssim_t = 0
        for ii in range(b):
            for jj in range(t):
                out_pic_p = out_pic[ii,jj, :, :]
                gt_t = gt[ii,jj, :, :]
                psnr_t += ski_psnr(gt_t,out_pic_p,data_range=1)
                ssim_t += ski_ssim(gt_t,out_pic_p,data_range=1)                
        psnr = psnr_t / (b*t)
        ssim = ssim_t / (b*t)
        psnr_list.append(np.round(psnr,4))
        ssim_list.append(np.round(ssim,4))
        out_list.append(out_rgb)
        gt_list.append(gt_rgb)

    for i,name in enumerate(test_data.data_list):
        _name,_ = name.split("_")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        img_dir = os.path.join(test_dir,_name)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for j in range(out.shape[0]):
            for k in range(out.shape[1]):
                img_path = img_dir +"/"+ str(j)+"_"+str(k)+"_"+str(psnr_list[i])+"_"+str(ssim_list[i])+".png"
                img = out[j,k,::]*255
                if args.color_channels == 3: img =img[:,:,::-1]
                img = img.astype(np.float32)
                cv2.imwrite(img_path, img)

    if logger is not None:
        logger.info("psnr_mean: {:.4f}.".format(np.mean(psnr_list)))
        logger.info("ssim_mean: {:.4f}.".format(np.mean(ssim_list)))
    return psnr_dict, ssim_dict

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    args = parse_args()

    if args.color_channels == 1:
        mask_path = './mask/gray_mask.mat'
    elif args.color_channels == 3:
        mask_path = './mask/color_mask.mat'
    mask = scio.loadmat(mask_path)['mask']
    mask = mask.transpose([2, 0, 1]).astype(np.float32)
    mask = mask[:args.B,:args.size[0],:args.size[1]]
    mask_s = np.sum(mask, axis=0)
    mask_s[mask_s == 0] = 1
    args.mask = torch.from_numpy(mask).float().to(args.device)
    args.mask_s = torch.from_numpy(mask_s).float().to(args.device)
    args.mask = einops.rearrange(args.mask,'t h w->1 t h w')
    args.mask_s = einops.rearrange(args.mask_s,'h w->1 1 h w')

    test_path = "test_results"
    network = HiSViT(dim=args.dim, frames=args.B, size=args.size, color_ch=args.color_channels, blocks=args.blocks).to(args.device)
    log_dir = os.path.join("test_results","log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    if args.test_weight_path is not None:
        logger.info('Loading pretrained model...')
        pretrained_dict = torch.load(args.test_weight_path)
        load_checkpoint(network, pretrained_dict, logger) 
    else:
        raise ValueError('Please offer a weight path for testing.')
    psnr_dict, ssim_dict = test(args, network, logger, test_path)
    logger.info("psnr: {}.".format(psnr_dict))
    logger.info("ssim: {}.".format(ssim_dict))
