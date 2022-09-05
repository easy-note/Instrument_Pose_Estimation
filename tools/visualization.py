import matplotlib.pyplot as plt 
from glob import glob
import os 
from PIL import Image
import pandas as pd 
import numpy as np
from post_processing import Post_Processing

import seaborn as sns
import matplotlib.pylab as plt
from utils.line_integral import get_points

class Visualization():
    def __init__(self, mode='test', dest_path=None):

        self.dest_path = dest_path
        if not os.path.exists(dest_path):
            os.makedirs(dest_path, exist_ok=True)
        self.regression_list = '/raid/datasets/public/EndoVisPose/Training/training/labels_jh/regression'
        # LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint (0,1,2,3,4)
        # joint1, joint2, joint1-joint2 pair id
        self.joint_pair_list = [(0, 2, 5), (1, 2, 6), (2, 3, 7), (3, 4, 8)]
        self.joint = {'LeftClasperPoint' : 0, 'RightClasperPoint' : 1, 'HeadPoint' : 2, 'ShaftPoint' : 3, 'EndPoint' : 4}
        
        self.joint_color = ['r', 'y', 'g', 'b', 'm']
        self.post_processing = Post_Processing()
    
    def list_load(self): # GT annotation 읽기 
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
    
    def regression_pp_list_load(self, img_path): # GT annotation 읽기 
        for filename in sorted(os.listdir(self.regression_list)):


            image = Image.open(os.path.join(img_path, filename.split('.')[0] + '.jpg'))
            regression = np.load(os.path.join(self.regression_list, filename))
            print(np.shape(regression))
            regression = np.transpose(regression, (2,0,1))
            regression = np.expand_dims(regression, axis=0)
            # self.show_heatmap(image, regression, filename.split('.')[0])
            label = self.post_processing.run(regression, 10)
            print(filename, np.expand_dims(label, axis=0), np.shape(regression))

            self.show(image, label, filename.split('.')[0])
    
    def show_heatmap(self, image, heatmap, title):
        data = np.sum(heatmap[:,:,:5], axis=2)
        plt.imshow(image)
        # ax = sns.heatmap(data, linewidth=0.3)
        plt.imshow(data, cmap='cool')
        plt.savefig(os.path.join(self.dest_path, title + '.png'))
        plt.close()

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
            # print(tool)
            if len(tool) == -5:
                continue
            for multiple_tools_idx in range(len(tool)):
                for point_idx in range(len(tool[multiple_tools_idx])):
                    point = tool[multiple_tools_idx][point_idx] 
                    if len(point) == 0:
                        continue
                    # x, y = point
                    y, x = point
             
                    if y == -1 or x == -1:
                        continue
                    plt.plot(x, y, color=self.joint_color[multiple_tools_idx], marker='o', markersize=5)
        
        # print(self.dest_path, title)
        plt.savefig(os.path.join(self.dest_path, title.split('.')[0] + '.png'))
        plt.close()

    def output_overlap(self, image, label, title):
        

    

if __name__ == '__main__':
    vis = Visualization(dest_path='/raid/results/pose/heatmap')
    # 'gt_path ='/raid/datasets/public/EndoVisPose/train_labels'
    vis.regression_pp_list_load('/raid/datasets/public/EndoVisPose/Training/training/image')
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