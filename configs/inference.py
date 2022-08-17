configs = dict()
# Defuat
configs['dest_path'] = '//raid/results/optimal_surgery/endovis/\
endovis/endovis_network/202208131354'


# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = dict()
dataset['normalization']['mean'] = [0.5, 0.5,0.5]#[0.3315324287939336, 0.30612040989419975, 0.42873558758384006]
dataset['normalization']['std'] = [0.5, 0.5,0.5]#[0.16884704188593896, 0.15812564825433872, 0.18358451252795385]


import os 
dataset['images'] = None #os.listdir('/raid/datasets/public/EndoVisPose/Training/test/image')
dataset['root'] = '/raid/datasets/public/EndoVisPose/Training/test/'

dataset['augmentation'] = dict() 
dataset['augmentation']['train'] = ['verticalflip', 'horizonflip']
dataset['augmentation']['val'] = []
dataset['augmentation']['test'] = []

dataset['img_size'] = (320, 256)

configs['nms'] = dict()
configs['nms']['window'] = 20

dataset['n_class'] = 9
dataset['num_parts'] = 5
dataset['num_connections'] = 4

configs['dataset'] = dataset


# Model
model = dict()
model['method'] = 'endovis_network'
model['n_channels'] = 64
model['bilinear'] = False
model['checkpoint'] = configs['dest_path'] + '/models/lastest.pth'
configs['model'] = model
