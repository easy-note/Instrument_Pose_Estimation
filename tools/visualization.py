import matplotlib.pyplot as plt 
from glob import glob
import os 
from PIL import Image
import pandas as pd 
import numpy as np

from utils.line_integral import get_points
class Visualization():
    def __init__(self, img_path=None, gt_path=None, pred_path=None, dest_path=None):
        self.img_path = img_path
        self.gt_path = gt_path 
        self.gt_list =  glob(os.path.join(gt_path, '*.json')) if gt_path is not None else None
        self.pred_path = pred_path 
        self.dest_path = dest_path

        # LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint (0,1,2,3,4)
        # joint1, joint2, joint1-joint2 pair id
        self.joint_pair_list = [(0, 2, 5), (1, 2, 6), (2, 3, 7), (3, 4, 8)]
        self.joint = {'LeftClasperPoint' : 0, 'RightClasperPoint' : 1, 'HeadPoint' : 2, 'ShaftPoint' : 3, 'EndPoint' : 4}
        
        self.joint_color = ['r', 'y', 'g', 'b', 'm']

    def list_load(self):
        for filename in self.gt_list:

            gt = pd.read_json(filename)
            for idx in range(len(gt['annotations'])):
     
                if not gt['annotations'][idx]:
                    continue
                name = gt['filename'][idx].split('/')[-1].split('.')[0] + '_{}.jpg'.format(filename.split('/')[-1].split('_')[0]) # img_000100_raw.png => img_000100_raw_train1.jpg
                image = Image.open(os.path.join(self.img_path, name))
                
                label = gt['annotations'][idx]
                yxlabel = [np.ones((5,2))*-1, np.ones((5,2))*-1]
    
                for tools in label:
                    if tools['class'] == 'TrackedPoint':
                        continue
                    if tools['id'] == 'tool1':
                        yxlabel[0][self.joint[tools['class']]] = int(tools['y']) , int(tools['x'])
                    else:
                        yxlabel[1][self.joint[tools['class']]] = int(tools['y']) , int(tools['x'])

                self.show(image, yxlabel, gt['filename'][idx].split('/')[-1].split('.')[0])

    def show(self, image, label, title):
        '''
        image HWC
        label 
        - [[(tool1_inst1_x, tool1_inst1_x), (tool1_inst2_x, tool1_inst2_x), .... ], [(tool2_inst1_x, tool1_inst2_x), ] ...]

        '''
        plt.imshow(image)
        plt.tight_layout()
        plt.axis('off')

        for tool in label:
            print(tool)
            if sum(sum(tool)) == -5:
                continue
            for point_idx in range(len(tool)):
                y, x = tool[point_idx] 
                if y == -1 or x == -1:
                    continue
                plt.plot(x, y, color=self.joint_color[point_idx], marker='o', markersize=5)
        plt.savefig(os.path.join(self.dest_path, title + '.png'))
        plt.close()

    

if __name__ == '__main__':
    vis = visualization(img_path='/raid/datasets/public/EndoVisPose/Training/training/image', gt_path ='/raid/datasets/public/EndoVisPose/train_labels', dest_path='/raid/results/pose')

    vis.list_load()
    # image = Image.open('/raid/datasets/public/EndoVisPose/Training/training/image/img_000190_raw_train1.jpg')
    # label = pd.read_json('/raid/datasets/public/EndoVisPose/train_labels/train1_labels_v2.json')['annotations'][189]
    # tmp = [np.ones((5,2))*-1, np.ones((5,2))*-1]
    # joint = {'LeftClasperPoint' : 0, 'RightClasperPoint' : 1, 'HeadPoint' : 2, 'ShaftPoint' : 3, 'EndPoint' : 4}
    # for tools in label:
    #     if tools['class'] == 'TrackedPoint':
    #         continue
    #     if tools['id'] == 'tool1':
    #         tmp[0][joint[tools['class']]] = int(tools['y']) , int(tools['x'])
    #     else:
    #         tmp[1][joint[tools['class']]] = int(tools['y']) , int(tools['x'])

    # vis.show(image, tmp, 'img_000190_raw_train1')