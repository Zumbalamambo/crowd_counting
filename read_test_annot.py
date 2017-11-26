# -*- coding: utf-8 -*-
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image


def convert_image(locs):
    im = np.zeros((1024,704))
    count = 0
    for loc in locs:
        count += 1
        im[int(loc[0])][int(loc[1])] = 255
    print count
    return im
        
path = 'data/original/shanghaitech/part_A_final/test_data/ground_truth/'
mat = io.loadmat(path+'GT_IMG_1.mat')

locs = mat['image_info'][0][0][0][0][0]

data = convert_image(locs)

im_read = Image.open('IMG_1.jpg')
#img = Image.fromarray(data, 'L')

#img.save('aa.jpg')



