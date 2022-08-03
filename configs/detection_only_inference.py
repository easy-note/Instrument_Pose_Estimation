configs = dict()
# Defuat
dest_path = '/raid/results/optimal_surgery/detection_only/endovis/detection_subnetwork/202208010841/'


# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = [[1,1,1],[1,1,1]]
dataset['batch_size'] = 32
dataset['shuffle'] = True
dataset['num_workers'] = 6
dataset['pin_memory'] = True

import os 
dataset['images'] = None #os.listdir('/raid/datasets/public/EndoVisPose/Training/test/image')
dataset['root'] = '/raid/datasets/public/EndoVisPose/Training/test/'

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
model['method'] = 'detection_subnetwork'
model['n_channels'] = 64
model['bilinear'] = False
model['checkpoint'] = '/raid/results/optimal_surgery/detection_only/endovis/detection_subnetwork/202208010841/models/student_lastest.pth'
configs['model'] = model
