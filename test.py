import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms

from models.UGSDA import UGSDA

torch.cuda.set_device(0)

mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

LOG_PARA = 100.0

dataRoots = [
    "./datasets/shanghaitech_part_A/test",
    "./datasets/shanghaitech_part_B/test",
    "./datasets/UCF-QNRF/test",
]

model_path = './pretrained_models/SHHA_parameters.pth'

def main():
    for dataRoot in dataRoots:
        file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]                                           
        test(file_list[0], model_path, dataRoot)
    

def test(file_list, model_path, dataRoot):

    net = UGSDA("UGSDA")
    net.cuda()
    net.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    net.eval()

    maes = AverageMeter()
    mses = AverageMeter()

    for filename in file_list:

        imgname = os.path.join(dataRoot, 'img', filename)
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'
        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)

        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)
        gt = np.sum(den)
        with torch.no_grad():
            img = Variable(img[None,:,:,:]).cuda()
            pred_map = net.test_forward(img)

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
        pred = np.sum(pred_map)/LOG_PARA

        maes.update(abs(gt - pred))
        mses.update(((gt - pred) * (gt - pred)))
    mae = maes.avg
    mse = np.sqrt(mses.avg)

    print(mae, mse)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count
if __name__ == '__main__':
    main()
