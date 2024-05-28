import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Network
from multi_read_data import MemoryFriendlyLoader
import time
parser = argparse.ArgumentParser("ruas")
parser.add_argument('--data_path', type=str, default='/home/suxin/mambaeec/data2/test/input/',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='/home/suxin/RUAS/result2/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='weights', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=7, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
test_low_data_names = args.data_path + '/*.jpg'
# test_low_data_names = r'H:\CVPR2021\LOL-700\input-100/*.png'
# test_low_data_names = r'H:\image-enhance\LLIECompared\DarkFace1000\input/*.png'


TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('gpu device = %d' % args.gpu)
    print("args = %s", args)

    model = Network()
    model = model.cuda()
    model_dict = torch.load('./ckpt/'+args.model+'.pt')
    model.load_state_dict(model_dict)


    for p in model.parameters():
        p.requires_grad = False
    # processing_time = 0
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('.')[0]
            # start_time = time.time()
            u_list, r_list = model(input)
            # end_time = time.time()
            # processing_time += (end_time - start_time)
            u_name = '%s.png' % (image_name).split('/')[-1]
            u_path = save_path + '/' + u_name
            print('processing {}'.format(u_name))
            
            if args.model == 'weights':
                save_images(u_list[-1], u_path)
            elif args.model == 'upe' or args.model == 'dark':
                save_images(u_list[-2], u_path)
            # print('processing time {}'.format(processing_time))


if __name__ == '__main__':
    main()
