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
        masks_dir = configs['masks_dir'][state]
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
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
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        img = self.load(img_file[0])
        label = self.load(mask_file[0])


        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']

        return img, [label.permute(2, 0, 1)]
