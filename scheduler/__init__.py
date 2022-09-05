import torch 

def get_optimizer(configs, model):
    optim_cofig = configs.optimization
    optim_method = optim_cofig['optim']
    lr = optim_cofig['init_lr']
    momentum = optim_cofig['momentum']
    weight_decay = optim_cofig['weight_decay']

    if optim_method == 'sgd':
        optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay)
            
    
    elif optim_method == 'adam':
        optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay)
#    ====================================================================================   

    scheduler_config = configs.scheduler
    scheduler_method = scheduler_config['scheduler']
                
    if scheduler_method == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=scheduler_config['lr_decay_epochs'], 
                    gamma=scheduler_config['gamma'], 
                    )
    elif scheduler_method == 'linear_lr':
        scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, 
                    start_factor= scheduler_config['start_factor'],
                    end_factor= scheduler_config['end_factor'],
                    total_iters= scheduler_config['total_iters'], 
                    last_epoch= scheduler_config['last_epoch'] 
                    )
    else:
        scheduler = None

    return optimizer, scheduler