import torch
from torch.utils.data import Dataset 
import scipy.io as scio
import h5py
import numpy as np
import logging 
import time 
import os
import cv2
import albumentations


class TrainData(Dataset):
    def __init__(self,args):
        self.data_dir= args.train_data_path
        self.data_list = os.listdir(args.train_data_path)
        self.img_files = []
        self.color_ch = args.color_channels
        self.ratio = args.B
        self.resize_h, self.resize_w = args.size
        self.mask = args.mask
        for image_dir in os.listdir(args.train_data_path):
            train_data_path = os.path.join(args.train_data_path,image_dir)
            data_path = os.listdir(train_data_path)
            data_path.sort()
            for sub_index in range(len(data_path)-self.ratio):
                sub_data_path = data_path[sub_index:]
                meas_list = []
                count = 0
                for image_name in sub_data_path:
                    meas_list.append(os.path.join(train_data_path,image_name))
                    if (count+1)%self.ratio==0:
                        self.img_files.append(meas_list)
                        meas_list = []
                    count += 1
        if self.color_ch == 3:
            r = np.array([[1, 0], [0, 0]])
            g1 = np.array([[0, 1], [0, 0]])
            g2 = np.array([[0, 0], [1, 0]])
            b = np.array([[0, 0], [0, 1]])
            self.rgb2raw = np.zeros([3, self.resize_h, self.resize_w])
            self.rgb2raw[0, :, :] = np.tile(r, (self.resize_h // 2, self.resize_w // 2))
            self.rgb2raw[1, :, :] = np.tile(g1, (self.resize_h // 2, self.resize_w // 2)) + np.tile(g2, (self.resize_h // 2, self.resize_w // 2))
            self.rgb2raw[2, :, :] = np.tile(b, (self.resize_h // 2, self.resize_w // 2))

    def __getitem__(self,index):
        image = cv2.imread(self.img_files[index][0])
        image_h, image_w = image.shape[:2]
        
        crop_h = np.random.randint(self.resize_h//2,image_h)
        crop_w = np.random.randint(self.resize_w//2,image_w)
        crop_p = np.random.randint(0,10)>5
        flip_p = np.random.randint(0,10)>5
        transform = albumentations.Compose([
            albumentations.CenterCrop(height=crop_h,width=crop_w,p=crop_p),
            albumentations.HorizontalFlip(p=flip_p),
            albumentations.Resize(self.resize_h,self.resize_w)
        ])
        rotate_flag = np.random.randint(0,10)>5

        gt = np.zeros([self.color_ch, self.ratio, self.resize_h, self.resize_w],dtype=np.float32)
        meas = np.zeros([self.resize_h, self.resize_w], dtype=np.float32)

        for i,image_path in enumerate(self.img_files[index]):
            image = cv2.imread(image_path)

            transformed = transform(image=image)
            image = transformed['image']

            if rotate_flag:
                image = cv2.flip(image, 1)
                image = cv2.transpose(image)
            if self.color_ch > 1:
                image = image.astype(np.float32) / 255.
                image = np.transpose(image, [2,0,1])[::-1,:,:]
                gt[:, i, :, :] = image
                raw = np.sum(self.rgb2raw*image, axis=0,keepdims=True)
            else:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)[:,:,0]
                image = image.astype(np.float32) / 255.
                gt[:, i, :, :] = image
        if self.color_ch == 1:
            return gt
        else:
            return gt, raw

    def __len__(self,):
        return len(self.img_files)

class TestData(Dataset):
    def __init__(self,args):
        self.data_path = args.test_data_path
        self.data_list = os.listdir(self.data_path)
        self.cr = args.B
        self.img_h, self.img_w = args.size
        self.mask = args.mask
        self.color_ch = args.color_channels
        if self.color_ch == 3:
            r = np.array([[1, 0], [0, 0]])
            g1 = np.array([[0, 1], [0, 0]])
            g2 = np.array([[0, 0], [1, 0]])
            b = np.array([[0, 0], [0, 1]])
            self.rgb2raw = np.zeros([3, self.img_h, self.img_w])
            self.rgb2raw[0, :, :] = np.tile(r, (self.img_h // 2, self.img_w // 2))
            self.rgb2raw[1, :, :] = np.tile(g1, (self.img_h // 2, self.img_w // 2)) + np.tile(g2, (self.img_h // 2, self.img_w // 2))
            self.rgb2raw[2, :, :] = np.tile(b, (self.img_h // 2, self.img_w // 2))

    def __getitem__(self,index):
        try:
            pic = scio.loadmat(os.path.join(self.data_path,self.data_list[index]))
            pic = pic['orig']
        except:
            pic = h5py.File(os.path.join(self.data_path,self.data_list[index]))
            pic = pic['orig']

        if self.color_ch == 1:
            pic = np.transpose(pic, [2, 0, 1])
        elif self.color_ch == 3:
            pic = np.transpose(pic, [0, 1, 3, 2])
            pic_rgb = np.zeros([pic.shape[0] // self.cr, 3, self.cr, self.img_h, self.img_w])
        else:
            raise ValueError('there is an error in the input color channel.')

        pic_gt = np.zeros([pic.shape[0] // self.cr, self.cr, self.img_h, self.img_w])
        
        for jj in range(pic.shape[0]):
            if jj // self.cr>=pic_gt.shape[0]:
                break
            if jj % self.cr == 0:
                n = 0
            if self.color_ch == 1:
                pic_t = pic[jj]
                pic_t = pic_t.astype(np.float32)
                pic_t = pic_t / 255.
            else:
                pic_c = pic[jj]
                pic_c = pic_c.astype(np.float32)
                pic_c = pic_c / 255.
                pic_rgb[jj // self.cr, :, n, :, :] = pic_c
                pic_t = np.sum(self.rgb2raw*pic_c, axis=0)

            pic_gt[jj // self.cr, n, :, :] = pic_t
            n += 1
        if self.color_ch == 1:
            return pic_gt
        else:
            return pic_gt, pic_rgb

    def __len__(self,):
        return len(self.data_list)


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def Logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s')

    localtime = time.strftime('%Y_%m_%d_%H_%M_%S')
    logfile = os.path.join(log_dir,localtime+'.log')
    fh = logging.FileHandler(logfile,mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 


def checkpoint(epoch, model, optimizer, model_out_path):
    torch.save({'pretrain_epoch':epoch,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()}, model_out_path)

def load_checkpoint(model, pretrained_dict, logger, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    load_dict = {k: p for k, p in pretrained_model_dict.items() if k in model_dict.keys()} # filtering parameters
    for k in load_dict: 
        if model_dict[k].shape != load_dict[k].shape:
            # logger.info("layer: {} parameters size is inconsistent! excpet {}, given {}".format(k, model_dict[k].shape, load_dict[k].shape))
            load_dict[k] = model_dict[k]    
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    logger.info('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict), len(load_dict)))

