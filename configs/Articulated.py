configs = dict()

# Dataset
dataset = dict()
dataset['dataset'] = 'endovis'
dataset['normalization'] = [[1,1,1],[1,1,1]]
dataset['batch_size'] = 256
dataset['shuffle'] = True
dataset['num_workers'] = 16
dataset['pin_memory'] = True

dataset['data_path'] = ''
dataset['n_class'] = 9
dataset['img_size'] = (256,320)

dataset['num_parts'] = 5
dataset['n_num_connectionsclass'] = 4

configs['dataset'] = dataset


# Optimization & Scheduler
optimization = dict()
optimization['optim'] = 'sgd'
optimization['momentum'] = 0.9
optimization['weight_decay'] = 1e-4
optimization['init_lr'] = 1e-2
optimization['epochs'] = 200

scheduler = dict()
scheduler['scheduler'] = 'step_lr'
scheduler['lr_decay_epochs'] = [60,120,150]
scheduler['gamma'] = 0.1

configs['optimization'] = optimization
configs['scheduler'] = scheduler

# Loss
loss = dict()

loss['method'] = ['binarycrossentropy', 'crossentropy']
loss['weight'] = [None, None]
loss['label_smoothing'] = [0.0, 0.0]


configs['loss'] = loss

# Model
model = dict()
model['method'] = 'endovis_model'

configs['model'] = model

# Metric
metric = dict()
metric['metric'] = 'rms' # root mean square
metric['threshold'] = 20
configs['metric'] = metric


# Defuat
configs['results'] = ''


