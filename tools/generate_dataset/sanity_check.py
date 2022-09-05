
import numpy as np
import os
import glob
import cv2

path = '/raid/dataset/public/general/EndoVisPose/Training/training/labels_jh/regression_10'
nps = glob.glob(os.path.join(path, '*.npy'))


for i in nps:
    save = './regression_10_img/{}'.format(i.split('/')[-1].split('.')[0])
    os.makedirs(save, exist_ok=True)

    x = np.load(i)
    
    for j in range(9):
        cv2.imwrite(os.path.join(save, '{}-{}.jpeg'.format(i.split('/')[-1], j)), x[:,:,j]*255)
    
    