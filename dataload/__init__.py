from .base import BasicDataset
from .carvan import CarvanaDataset
from .endovis import EndovisDataset
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch as AP


__all__ = [
    'BasicDataset', 'CarvanaDataset'
]
'''
dataset['img_size'] = (256,320)
'''
def get_dataloaders(configs):
    configs = configs['dataset'] 
    
    
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
    trans = [A.Resize(width=width, height=height )]

    # for method in configs['augmentation'][mode]:
    #     if method == 'verticalflip':
    #         trans.append(A.VerticalFlip(0.7))
    #     elif method == 'horizonflip':
    #         trans.append(A.HorizontalFlip(0.7))
    if mode == 'train':
        trans.append(A.RandomResizedCrop(height=height, width=width, scale=(0.7, 0.8)))
        trans.append(A.ShiftScaleRotate(scale_limit=[-0.3, 0.35], rotate_limit=[-45,45]))


    
        trans.append(A.OneOf([
            A.Blur(),
            A.RandomBrightnessContrast(p=0.2)
        ]))
        trans.append(A.OneOf([
            A.VerticalFlip(p=1)
            A.HorizontalFlip(p=1)
        ]))
    
    mean, std = configs['normalization']['mean'], configs['normalization']['std']
    trans.append(A.Normalize(mean=mean, std=std))
    trans.append(AP.transforms.ToTensorV2())
    return trans