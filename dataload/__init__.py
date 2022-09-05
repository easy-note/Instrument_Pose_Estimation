from .base import BasicDataset
from .carvan import CarvanaDataset
from .endovis import EndovisDataset
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch as AP


__all__ = [
    'BasicDataset', 'CarvanaDataset', 'EndovisDataset'
]
'''
dataset['img_size'] = (256,320)
'''
def get_dataloaders(configs):
     
    
    train_transform = A.Compose(get_augmentation(configs, 'train'), p=1 )
    val_transform = A.Compose(get_augmentation(configs, 'val'), p=1)
    test_transform = A.Compose(get_augmentation(configs, 'test'), p=1)

    if configs['dataset'] == 'endovis':
        train_set = EndovisDataset(configs, state='train', transforms = train_transform)
        valid_set = EndovisDataset(configs, state='val', transforms = val_transform)
        test_set = EndovisDataset(configs, state='test', transforms = test_transform)
    train_loader = DataLoader(
                train_set,
                batch_size=configs['batch_size'],
                num_workers=configs['num_workers'],
                shuffle=True,
                drop_last=False,
                )

    val_loader = DataLoader(
                valid_set,
                batch_size=configs['batch_size'],
                num_workers=configs['num_workers'],
                drop_last=False,
                )
    
    test_loader = DataLoader(
                test_set,
                batch_size=configs['batch_size'],
                num_workers=configs['num_workers'],
                drop_last=False,
                )

    return train_loader, val_loader, test_loader


def get_augmentation(configs, mode):

    width, height = configs['img_size']
    trans = [A.Resize(height=height, width=width)]

    if mode == 'train':
        trans.append(A.HorizontalFlip(p=0.5))
        trans.append(A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=[-45,45]))
        
        # trans.append(A.OneOf([
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        # ], p=1))
        # trans.append(A.OneOf([
        #     A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     A.GridDistortion(p=0.5),
        #     A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        # ], p=0.8))

    mean, std = configs['normalization']['mean'], configs['normalization']['std']
    trans.append(A.Normalize(mean=mean, std=std))
    trans.append(AP.transforms.ToTensorV2())
    return trans