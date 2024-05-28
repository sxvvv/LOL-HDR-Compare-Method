import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
import time
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='/home/suxin/mambaeec/data2/test/input',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results2/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='/home/suxin/SCI/weights/easy.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=7, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

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

    model = Finetunemodel(args.model)
    model = model.cuda()
    total_time = 0
    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('\\')[-1].split('.')[0]
            start_time = time.time()
            i, r = model(input)
            end_time = time.time()
            total_time += end_time - start_time
            u_name = '%s.jpg' % (image_name).split('/')[-1]
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(r, u_path)

        print(total_time)

if __name__ == '__main__':
    main()
