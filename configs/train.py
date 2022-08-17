configs = dict()

# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = dict()
dataset['normalization']['mean'] =[0.5, 0.5, 0.5]
dataset['normalization']['std'] = [0.5, 0.5, 0.5]
dataset['batch_size'] = 3
dataset['shuffle'] = True
dataset['num_workers'] = 6
dataset['pin_memory'] = True

dataset['images_dir'] = {
    'train': '/raid/datasets/public/EndoVisPose/Training/training/image', 
    'val': '/raid/datasets/public/EndoVisPose/Training/test/image',
    'test': '/raid/datasets/public/EndoVisPose/Training/test/image'}
dataset['detection_mask'] = {
    'train': '/raid/datasets/public/EndoVisPose/Training/training/labels_jh/segmentation', 
    'val': '/raid/datasets/public/EndoVisPose/Training/test/labels_jh/segmentation',
    'test': '/raid/datasets/public/EndoVisPose/Training/test/labels_jh/segmentation'}
dataset['regression_mask'] = {
    'train': '/raid/datasets/public/EndoVisPose/Training/training/labels_jh/regression', 
    'val': '/raid/datasets/public/EndoVisPose/Training/test/labels_jh/regression',
    'test': '/raid/datasets/public/EndoVisPose/Training/test/labels_jh/regression'}

dataset['augmentation'] = dict() 
dataset['augmentation']['train'] = ['verticalflip', 'horizonflip']
dataset['augmentation']['val'] = []
dataset['augmentation']['test'] = []

dataset['img_size'] = (320, 256)

dataset['n_class'] = 9
dataset['num_parts'] = 5
dataset['num_connections'] = 4

configs['dataset'] = dataset

configs['nms'] = dict()
configs['nms']['window'] = 20

# Optimization & Scheduler
optimization = dict()
optimization['optim'] = 'adam'
optimization['momentum'] = 0.98
optimization['weight_decay'] = 1e-5
optimization['init_lr'] = 0.01
optimization['epochs'] = 100

scheduler = dict()
scheduler['scheduler'] = 'linear_lr' #'step_lr' LinearLR
scheduler['start_factor'] = 1.0
scheduler['end_factor'] = 0
scheduler['total_iters'] = optimization['epochs']
scheduler['last_epoch'] = -1
scheduler['gamma'] = 0.1 # 0.05

configs['optimization'] = optimization
configs['scheduler'] = scheduler

# Loss
loss = dict()

loss['method'] = ['bce', 'mse']
loss['activation'] = [None, None]
loss['reduction'] = 'mean'
loss['weight'] = [None, None]#[ [0.1, 0.9], [0.1, 0.9]]
loss['label_smoothing'] = [0.0, 0.0]
configs['loss'] = loss

# Model
model = dict()
model['method'] = 'endovis_network'
model['n_channels'] = 64
model['bilinear'] = False
configs['model'] = model

# Metric
metric = dict()
metric['metric'] = 'rms' # root mean square
metric['threshold'] = 20
configs['metric'] = metric


# Defuat
configs['results'] = '/raid/results/optimal_surgery/endovis'

