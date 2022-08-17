import os 
from PIL import Image
import cv2 
from multiprocessing import Pool
import numpy as np 
def rgb( filename):
    root = '/raid/datasets/public/EndoVisPose/Training/training/image'
    img = Image.open(os.path.join(root, filename))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    b, g, r = [],[],[]
    for i in range(len(img)):
        for j in range(len(img[0])):
            b.append(img[i,j,0])
            g.append(img[i,j,1])
            r.append(img[i,j,2])
    return (np.mean(b), np.std(b), len(b)) ,(np.mean(g), np.std(g), len(g)) ,(np.mean(r), np.std(r), len(r))
if __name__ == '__main__':
    root = '/raid/datasets/public/EndoVisPose/Training/training/image'

    # for filename in os.listdir(root):
    pool = Pool(processes = 100)
    result = pool.map(rgb, os.listdir(root))
    pool.close()
    pool.join()
    print(np.shape(result))
    b_mean_list, b_std_list= [], []
    g_mean_list, g_std_list= [], []
    r_mean_list, r_std_list= [], []


    for i in range(len(result)):
        b_mean, b_std, b_len = result[i][0]
        b_mean_list.append(b_mean)
        b_std_list.append(b_std)

        g_mean, g_std, g_len = result[i][1]
        g_mean_list.append(g_mean)
        g_std_list.append(g_std)
        
        r_mean, r_std, r_len = result[i][2]
        r_mean_list.append(r_mean)
        r_std_list.append(r_std)

    print(np.mean(b_mean_list) / 255, np.mean(b_std_list) / 255)

    print(np.mean(g_mean_list) / 255, np.mean(g_std_list) / 255)

    print(np.mean(r_mean_list) / 255, np.mean(r_std_list) / 255)