import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import cv2

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

'''dataset['dataset'] = 'endovis'
dataset['normalization'] = [[1,1,1],[1,1,1]]
dataset['batch_size'] = 32
dataset['shuffle'] = True
dataset['num_workers'] = 6
dataset['pin_memory'] = True
'''
class EndovisDataset(Dataset):
    def __init__(self, configs,  state: str = 'train', transforms=None):
        images_dir = configs['images_dir'][state]
        detection_mask_dir = configs['detection_mask'][state] 
        regression_mask_dir = configs['regression_mask'][state] 
        self.images_dir = Path(images_dir)
        self.detection_mask_dir = Path(detection_mask_dir)
        self.regression_mask_dir = Path(regression_mask_dir)
        self.transforms = transforms

        self.ids = [splitext(file)[0] for file in sorted(listdir(images_dir)) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)

  
    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            # return Image.fromarray(np.load(filename))
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return cv2.cvtColor(np.array(Image.open(filename)), cv2.COLOR_RGB2BGR)

    def __getitem__(self, idx):
        name = self.ids[idx]
        detection_file = list(self.detection_mask_dir.glob(name + '.*'))
        regression_file = list(self.regression_mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(detection_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {detection_file}'
        assert len(regression_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {regression_file}'

        img = self.load(img_file[0])
        detection_mask = self.load(detection_file[0])
        regression_mask = self.load(regression_file[0])
        if self.transforms is not None:
            transformed = self.transforms(image=img, masks=[detection_mask, regression_mask])
            img = transformed['image']
            detection_mask = transformed['masks'][0]  # + np.random.uniform(low=.01, high=.01)
            regression_mask = transformed['masks'][1]  #+ np.random.uniform(low=-.01, high=.01)

        return img.float(), [detection_mask.permute(2, 0, 1) , regression_mask.permute(2, 0, 1)]
