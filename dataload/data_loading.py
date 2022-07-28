import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        newW, newH = 256, 320
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            # return Image.fromarray(np.load(filename))
            return Image.fromarray(np.load(filename)[:,:,0]), Image.fromarray(np.load(filename)[:,:,1]), Image.fromarray(np.load(filename)[:,:,2]), Image.fromarray(np.load(filename)[:,:,3]), Image.fromarray(np.load(filename)[:,:,4]), Image.fromarray(np.load(filename)[:,:,5]), Image.fromarray(np.load(filename)[:,:,6]), Image.fromarray(np.load(filename)[:,:,7]), Image.fromarray(np.load(filename)[:,:,8])
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        # mask = self.preprocess(mask, self.scale, is_mask=True)

        mask0 = self.preprocess(mask0, self.scale, is_mask=True)
        mask1 = self.preprocess(mask1, self.scale, is_mask=True)
        mask2 = self.preprocess(mask2, self.scale, is_mask=True)
        mask3 = self.preprocess(mask3, self.scale, is_mask=True)
        mask4 = self.preprocess(mask4, self.scale, is_mask=True)
        mask5 = self.preprocess(mask5, self.scale, is_mask=True)
        mask6 = self.preprocess(mask6, self.scale, is_mask=True)
        mask7 = self.preprocess(mask7, self.scale, is_mask=True)
        mask8 = self.preprocess(mask8, self.scale, is_mask=True)

        mask = np.stack((mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8), axis=0)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
