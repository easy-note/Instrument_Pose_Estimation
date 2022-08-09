configs = dict()

# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = [[1,1,1],[1,1,1]]
dataset['batch_size'] = 8
dataset['shuffle'] = True
dataset['num_workers'] = 6
dataset['pin_memory'] = True

dataset['images_dir'] = {
    'train': '/raid/dataset/public/general/EndoVisPose/Training/training/image', 
    'val': '/raid/dataset/public/general/EndoVisPose/Training/test/image',
    'test': '/raid/dataset/public/general/EndoVisPose/Training/test/image'}
dataset['seg_masks_dir'] = {
    'train': '/raid/dataset/public/general/EndoVisPose/Training/training/labels_jh/segmentation', 
    'val': '/raid/dataset/public/general/EndoVisPose/Training/test/labels_jh/segmentation',
    'test': '/raid/dataset/public/general/EndoVisPose/Training/test/labels_jh/segmentation'}
dataset['regress_masks_dir'] = {
    'train': '/raid/dataset/public/general/EndoVisPose/Training/training/labels_jh/regression', 
    'val': '/raid/dataset/public/general/EndoVisPose/Training/test/labels_jh/regression',
    'test': '/raid/dataset/public/general/EndoVisPose/Training/test/labels_jh/regression'}

dataset['augmentation'] = dict() 
dataset['augmentation']['train'] = ['verticalflip', 'horizonflip']
dataset['augmentation']['val'] = []
dataset['augmentation']['test'] = []

dataset['img_size'] = (256,320)

dataset['n_class'] = 9
dataset['num_parts'] = 5
dataset['num_connections'] = 4

configs['dataset'] = dataset

configs['nms'] = dict()
configs['nms']['window'] = 200

# Optimization & Scheduler
optimization = dict()
optimization['optim'] = 'sgd'
optimization['momentum'] = 0.98
optimization['weight_decay'] = 1e-4
optimization['init_lr'] = 0.001
optimization['epochs'] = 100

scheduler = dict()
scheduler['scheduler'] = 'step_lr'
scheduler['lr_decay_epochs'] = 10
scheduler['gamma'] = 0.5

configs['optimization'] = optimization
configs['scheduler'] = scheduler

# Loss
loss = dict()

loss['method'] = ['bce', 'mse']
loss['activation'] = ['sigmoid', None]
loss['reduction'] = 'mean'
loss['weight'] = [ None]
loss['label_smoothing'] = [0.0]
configs['loss'] = loss

# Model
model = dict()
# model['method'] = 'detection_subnetwork'
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
configs['results'] = '/raid/users/cv_ljh_0/instrument_pose/models'


