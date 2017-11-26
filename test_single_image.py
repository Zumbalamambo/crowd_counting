import os
import torch
import numpy as np
import time
import cv2
import sys
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils

def read_image(img_path):
    if sys.argv[1]:
	img_path = str(sys.argv[1])
    img = cv2.imread(img_path,0)
    img = img.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = (ht/4)*4
    wd_1 = (wd/4)*4
    img = cv2.resize(img,(wd_1,ht_1))
    img = img.reshape((1,1,img.shape[0],img.shape[1]))
    return img

"""Enable cudnn"""
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

"""Directories"""
data_path =  './input/IMG_13.jpeg'

model_path = './saved_models/mcnn_shtechA_1998.h5'

output_dir = './output/'

net = CrowdCounter()

"""Load trained model"""      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
gt_data = None

"""Load image from data_path"""      
im_data = read_image(data_path)

"""Calculate density map"""      
density_map = net(im_data, gt_data)

"""convert to numpy array"""      
density_map = density_map.data.cpu().numpy()

"""estimation count from density map"""      
et_count = np.sum(density_map)

"""Save result: density map; stack input image and density map"""
utils.save_results(im_data, int(et_count), density_map, output_dir)
utils.save_density_map(density_map, output_dir)


import matplotlib.pyplot as plt	
img = cv2.imread('./output/results.png')
plt.imshow(img, cmap='gray')
plt.show()
print et_count
