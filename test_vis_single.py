import sys
import getopt
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms

from PIL import Image
from models.UGSDA import UGSDA
from matplotlib import pyplot as plt


arguments_intDevice = 0
arguments_strModel = "UGSDA"
arguments_strModelStateDict = './pretrained_models/SHHB_parameters.pth'
arguments_strImg = './images/SHHB_vis3.jpg'
arguments_strOut = './outputs/SHHB_vis3.png'

# do not need to change
mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--device' and strArgument != '': arguments_intDevice = int(strArgument)  # device number
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # model type
    if strOption == '--model_state' and strArgument != '': arguments_strModelStateDict = strArgument  # path to the model state
    if strOption == '--img_path' and strArgument != '': arguments_strImg = strArgument  # path to the image
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored

torch.cuda.set_device(arguments_intDevice)

def test_vis_single(img_path, save_path):
    if arguments_strModel == "UGSDA":
        net = UGSDA(arguments_strModel)
        net.cuda()
        net.load_state_dict(torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage), strict=False)
        net.eval()

    else:
        raise ValueError('Network cannot be recognized. Please define your own Network here.')
    
    img = Image.open(img_path)

    if img.mode == 'L':
        img = img.convert('RGB')

    img = img_transform(img)

    with torch.no_grad():
        img = Variable(img[None,:,:,:]).cuda()
        pred_map = net.test_forward(img)

    pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
    pred = np.sum(pred_map)/100.0
    pred_map = pred_map/np.max(pred_map+1e-20)

    print("count result is {}".format(pred))

    den_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    den_frame.axes.get_yaxis().set_visible(False)
    den_frame.axes.get_xaxis().set_visible(False)
    den_frame.spines['top'].set_visible(False) 
    den_frame.spines['bottom'].set_visible(False) 
    den_frame.spines['left'].set_visible(False) 
    den_frame.spines['right'].set_visible(False) 
    plt.savefig(save_path, bbox_inches='tight',pad_inches=0,dpi=150)
    plt.close()

    print("save pred density map in {} success".format(arguments_strOut))

    print("end")

if __name__ == '__main__':
    test_vis_single(arguments_strImg, arguments_strOut)
