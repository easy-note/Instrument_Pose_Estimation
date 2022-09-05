
# Defuat
dest_path = '//raid/results/optimal_surgery/endovis/\
endovis/endovis_network/202208300920/'

# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = {'mean': [0.5, 0.5,0.5], 'std' : [0.5, 0.5,0.5]}
#[0.3315324287939336, 0.30612040989419975, 0.42873558758384006]
#[0.16884704188593896, 0.15812564825433872, 0.18358451252795385]
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

dataset['img_size'] = (320,256)

dataset['n_class'] = 9
dataset['num_parts'] = 5
dataset['num_connections'] = 4


# Post processing
post_process = dict()
post_process['nms'] = dict()
post_process['nms']['window'] = 100

# Loss
loss = dict()

loss['method'] = ['bce']
loss['activation'] = 'sigmoid'
loss['reduction'] = 'mean'
loss['weight'] = [ None]
loss['label_smoothing'] = [0.0]

# Model
model = dict()
model['method'] = 'endovis_network'
model['n_channels'] = 64
model['bilinear'] = False
model['checkpoint'] = dest_path + '/models/lastest.pth'

# Metric
metric = dict()
metric['metric'] = 'rms' # root mean square
metric['threshold'] = 20

