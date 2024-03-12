import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
import importlib
import time
import psutil




def train_net(config, device, epochs_wanted):
    writer_path = ''.join(['experiment', '/',config.model_kwargs["model_class"],'/','runs/', config._name_config])
    writer = SummaryWriter(writer_path)

    device = torch.device(device)

    # load the parameters of the model and put it on the gpu if available
    model = load_pmodel(config, device)
    model = model.to(device)

    # loaders
    train_dtst, eval_dtst = get_datasets(config)

    training_loader = DataLoader(train_dtst, batch_size=1, shuffle=True)
    eval_loader = DataLoader(eval_dtst, batch_size=1, shuffle=False)

    # Training components needed
    loss_func = get_loss_function(config)
    optimizer = get_optimizer(config, model)


    last_epoch = config.last_epoch
    for epoch in range(last_epoch, last_epoch + epochs_wanted):
        print (f"\n ...\nEPOCH: {epoch+1} starting \n ...\n")

        # Train for the epoch
        model.train(True)
        avg_train_loss = train_1_epoch(model,training_loader=training_loader, loss_func=loss_func, optimizer=optimizer)

        # Set the model to evaluation mode, disabling dropout 
        model.eval()
        running_eval_loss = 0.0
        # Disable gradient computation and reduce memory consumption with torch.no_grad()
        with torch.no_grad():
            print(f"Evaluation starting for epoch : {epoch}")
            start = time.time()
            for eval_data in eval_loader:
                # Get data from evaluation dataset
                vinputs, vlabels = eval_data
                vinputs = vinputs[0].to(device)
                vlabels = vlabels[0].to(device)
                # Model estimation
                voutputs = model(vinputs)
                # Loss computations
                eval_loss = loss_func(voutputs, vlabels)
                running_eval_loss += eval_loss
        avg_eval_loss = running_eval_loss / len(eval_loader)
        print(f"TRAIN loss: {avg_train_loss} | EVAL loss: {avg_eval_loss} for EPOCH: {epoch+1}\n")

        end = time.time()
        print(f"Evaluation time for epoch {epoch} is : {end-start}")
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_train_loss, 'Validation' : avg_eval_loss },
                        epoch + 1)
        writer.flush()

        # Update config JSON with new epoch number
        config.last_epoch += 1
        # Track best performance, and save the model's state
        if avg_eval_loss < config.best_eval_loss:
            # Update best loss and model
            config.best_eval_loss = avg_eval_loss.item()
            # Update the config file for new best model
            config.epoch_best_model = config.last_epoch
            new_model_path = ''.join(['experiment', '/',config.model_kwargs["model_class"],'/','saved_models', '/',config._name_config, '/','epoch_', str(config.epoch_best_model), '.pt'])
            torch.save(model.state_dict(), new_model_path)

        # Save configuration file
        config.save()
        #save last model
        new_model_path = ''.join(['experiment', '/',config.model_kwargs["model_class"],'/','saved_models', '/',config._name_config, '/','lastmodel.pt'])
        torch.save(model.state_dict(), new_model_path)
    return model





def load_pmodel(config, device):

    model_path = ''.join(['experiment', '/',config.model_kwargs["model_class"],'/','saved_models/',config._name_config, '/epoch_', str(config.epoch_best_model), '.pt'])

    model_class = get_model_class(config)
    model = model_class(**config.model_kwargs)

    # If model doesn't exist at location, initialize it
    if not os.path.isfile(model_path):

        dir_path = ''.join(['experiment', '/',config.model_kwargs["model_class"],'/','saved_models/',config._name_config])
        os.makedirs(dir_path)

        print(f"Model weights for {model._get_name()} starting initialization\n ...")
        model = init_net(model)
        print(f"Model weights for {model._get_name()} finished initialization\n")

        print(f"Model weights saving at {model_path} \n ...")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved at {model_path} \n")
    
    else:
        print(f"Model weights for {model_path} loading \n...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights for {model_path} loaded\n")

    return model


def get_model_class(config):

    model_module = importlib.import_module(config.model_kwargs["model_module"])
    model_class = getattr(model_module, config.model_kwargs["model_class"])
    return model_class


def get_datasets(config):

    train_module = importlib.import_module(config.dtst_train_kwargs["dtst_module"])
    eval_module = importlib.import_module(config.dtst_eval_kwargs["dtst_module"])

    train_class = getattr(train_module, config.dtst_train_kwargs["dtst_class_name"])
    eval_class = getattr(eval_module, config.dtst_eval_kwargs["dtst_class_name"])

    try:
        train_dtst = train_class(train_dir= config._train_dir, **config.dtst_train_kwargs)
        eval_dtst = eval_class(train_dir= config._eval_dir, **config.dtst_eval_kwargs)
    except Exception as e:
        raise e

    return train_dtst, eval_dtst

def get_optimizer(config, model):

    if config.optimizer_args['optim_mode'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer_args["_learning_rate"])

    elif config.optimizer_args['optim_mode'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.optimizer_args["_learning_rate"], momentum=config.optimizer_args["_momentum"])

    else:
        raise NotImplementedError(f"mode {config.optimizer_args['mode']} is not implemented")

    return optimizer

def get_loss_function(config):

    if config.loss_type == 'custom':
        loss_func_module = importlib.import_module(config.loss_func_kwargs["loss_func_module"])
        loss_func_class = getattr(loss_func_module, config.loss_func_kwargs["loss_func_class"])

        loss_func = loss_func_class(**config.loss_func_kwargs)
        return loss_func
    elif config.loss_type == 'standard':

        match config.loss_func:
            case 'NLLLoss':
                loss_func = nn.NLLLoss(weight=config.loss_func_std_args['weight'], reduction=config.loss_func_std_args['reduction'])

        return loss_func
    else:
        raise NotImplemented(f"Mode {config.loss_type} is not implemented. Try 'custom' or 'standard'.")

def get_device():

    # chooses between nvidia gpu, apple silicon gpu or cpu
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'

    return device


def train_1_epoch(model, training_loader, loss_func, optimizer):
    
    tot_loss = 0.
    dev = next(model.parameters()).device

    total_memory_ised = 0

    start = time.time()

    print(f"There is {len(training_loader)} batches for this epoch")
    for idx, data in enumerate(training_loader):

        
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs[0].to(dev)
        labels = labels[0].to(dev)
        

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients

        loss = loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        tot_loss += loss.item()

        # get memory usage
        if 'mps' in str(dev):
            total_memory_ised += torch.mps.current_allocated_memory()/1e6

        elif 'cuda' in str(dev):
            total_memory_ised += torch.cuda.memory_allocated()/1e6
        else:
            total_memory_ised += psutil.Process(os.getpid()).memory_info()[0]/(2.**30)



    end = time.time()
    print(f"Training for {idx} batches takes: {end-start}")
    average_memory = total_memory_ised/idx
    print(f"Average memory usage of :{average_memory}")
    

    return tot_loss / len(training_loader)




def init_net (model):

    # from :https://pytorch.org/docs/stable/nn.init.html
    torch.no_grad()
    model.apply(init_weights)
    
    return model
    


def init_weights(layer):

    # Initialize weights and biases
    if isinstance(layer, nn.Conv1d):
        nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.01))

    if isinstance(layer, nn.Linear) and layer.weight.data.shape[0] == 1:
        nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('sigmoid'))

    if isinstance(layer, nn.Linear) and layer.weight.data.shape[0] != 1:
        nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('tanh'))


