import argparse 
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument('--train_data_path',type=str, default='./data/traindata/DAVIS/JPEGImages/480p')
    parser.add_argument("--test_weight_path", default='./checkpoint/hisvit13_gray.pth', help='hisvit9_gray or hisvit13_gray or hisvit9_color', type=str)
    parser.add_argument('--test_data_path',type=str, default='./data/testdata/gray_256', help='gray_256 or color_512') 
    parser.add_argument('--B', default=8, type=int)
    parser.add_argument('-l','--size', default=[256,256], type=json.loads, help='[256,256] or [512,512]')
    parser.add_argument('--color_channels', default=1, help='1 [gray] or 3 [color]', type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--dim', default=[128, 256, 128])    
    parser.add_argument('--blocks', default=13, type=int)      
    parser.add_argument('--save_model_step', default=1, type=int)
    parser.add_argument('--save_train_image_step', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iter_step', default=200, type=int)
    parser.add_argument('--test_flag', default=False, type=bool)
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()
    return args
