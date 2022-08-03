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
    
    
    train_transform = A.Compose(get_augmentation(configs, 'train'))
    val_transform = A.Compose(get_augmentation(configs, 'val'))
    test_transform = A.Compose(get_augmentation(configs, 'test'))

    if configs['dataset'] == 'endovis':
        train_set = EndovisDataset(configs, transforms = train_transform)
        valid_set = EndovisDataset(configs, transforms = val_transform)
        test_set = EndovisDataset(configs, transforms = test_transform)
    train_loader = DataLoader(
                train_set,
                batch_size=configs['batch_size'],
                num_workers=configs['num_workers'],
                shuffle=True,
                drop_last=True,
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

    return train_loader, val_loader, test_set


def get_augmentation(configs, mode):

    width, height = configs['img_size']
    trans = [A.Resize(width=width, height=height )]

    for method in configs['augmentation'][mode]:
        if method == 'verticalflip':
            trans.append(A.VerticalFlip(0.5))
        elif method == 'horizonflip':
            trans.append(A.HorizontalFlip(0.5))

    
    
    trans.append(A.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))
    trans.append(AP.transforms.ToTensorV2())
    return trans