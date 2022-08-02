configs = dict()

# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = [[1,1,1],[1,1,1]]
dataset['batch_size'] = 32
dataset['shuffle'] = True
dataset['num_workers'] = 6
dataset['pin_memory'] = True


dataset['images_dir'] = {
    'train': '/raid/datasets/public/EndoVisPose/Training/training/image', 
    'val': '/raid/datasets/public/EndoVisPose/Training/test/image',
    'test': '/raid/datasets/public/EndoVisPose/Training/test/image'}
dataset['masks_dir'] = {
    'train': '/raid/datasets/public/EndoVisPose/Training/training/label', 
    'val': '/raid/datasets/public/EndoVisPose/Training/test/label',
    'test': '/raid/datasets/public/EndoVisPose/Training/test/label'}

dataset['augmentation'] = dict() 
dataset['augmentation']['train'] = ['verticalflip', 'horizonflip']
dataset['augmentation']['val'] = []
dataset['augmentation']['test'] = []

dataset['img_size'] = (256,320)

dataset['n_class'] = 9
dataset['num_parts'] = 5
dataset['num_connections'] = 4

configs['dataset'] = dataset



# Loss
loss = dict()

loss['method'] = ['bce']
loss['activation'] = 'sigmoid'
loss['reduction'] = 'mean'
loss['weight'] = [ None]
loss['label_smoothing'] = [0.0]
configs['loss'] = loss

# Model
model = dict()
model['method'] = 'detection_subnetwork'
model['n_channels'] = 64
model['bilinear'] = False
model['checkpoint'] = ''
configs['model'] = model

# Metric
metric = dict()
metric['metric'] = 'rms' # root mean square
metric['threshold'] = 20
configs['metric'] = metric


# Defuat
configs['results'] = '/raid/results/optimal_surgery/detection_only'


configs['results'] = '/raid/results/optimal_surgery/detection_only'


