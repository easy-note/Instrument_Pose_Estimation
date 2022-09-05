# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = dict()
dataset['normalization']['mean'] =[0.5, 0.5, 0.5]
dataset['normalization']['std'] = [0.5, 0.5, 0.5]
dataset['batch_size'] = 5
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


# Loss
loss = dict()

loss['method'] = ['bce', 'mse']
loss['activation'] = [None, None]
loss['reduction'] = 'mean'
loss['weight'] = [None, None]#[ [0.1, 0.9], [0.1, 0.9]]
loss['label_smoothing'] = [0.0, 0.0]


# Model
model = dict()
model['method'] = 'endovis_network'
model['n_channels'] = 64
model['bilinear'] = False


# Metric
metric = dict()
metric['metric'] = 'rms' # root mean square
metric['threshold'] = 20



# Post processing
post_process = dict()
post_process['nms'] = dict()
post_process['nms']['window'] = 20

# Defuat
results = '/raid/results/optimal_surgery/endovis'

from datetime import datetime
date = datetime.today().strftime("%Y%m%d%H%M") 

project_name = 'G_IPE_CV'
run_name = "baseline_except_vertical_flip_{}".format(date)


import math

sweep_config = {
    'project' : 'G_IPE_CV',
    'entity': 'vision_ai',
    'program': 'train.py',
    'metric' : {
        'name': 'val_loss',
        'goal': 'minimize'   
        },
    'parameters' : {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
 
        'epochs': {
            'values': [50, 100]
            },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
            },
        'batch_size': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(32),
            'max': math.log(256),
            }
        },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }
