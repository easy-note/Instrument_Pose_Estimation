configs = dict()
# Defuat
dest_path = '/raid/users/cv_ljh_0/instrument_pose/models/endovis/endovis_network/202208090658/'


# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = [[1,1,1],[1,1,1]]
dataset['batch_size'] = 32
dataset['shuffle'] = True
dataset['num_workers'] = 6
dataset['pin_memory'] = True

import os 
# dataset['images'] = None #os.listdir('/raid/datasets/public/EndoVisPose/Training/test/image')
dataset['images'] = os.listdir('/raid/dataset/public/general/EndoVisPose/Training/test/image')
dataset['root'] = '/raid/dataset/public/general/EndoVisPose/Training/test/image'

dataset['augmentation'] = dict() 
dataset['augmentation']['train'] = ['verticalflip', 'horizonflip']
dataset['augmentation']['val'] = []
dataset['augmentation']['test'] = []

dataset['img_size'] = (256,320)

dataset['n_class'] = 9
dataset['num_parts'] = 5
dataset['num_connections'] = 4

configs['dataset'] = dataset


# Model
model = dict()
model['method'] = 'endovis_network'
model['n_channels'] = 64
model['bilinear'] = False
model['checkpoint'] = '/raid/users/cv_ljh_0/instrument_pose/models/endovis/endovis_network/202208090658/models/model_lastest.pth'
configs['model'] = model