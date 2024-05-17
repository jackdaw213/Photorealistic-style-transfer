import torch
import tqdm as tq
import torchmetrics
from datetime import datetime
import wandb

import utils

def train_style(model, optimizer, scaler, loss_func, loader, device, args):

    model.train()

    content_metric = torchmetrics.MeanMetric().to(device)
    style_metric = torchmetrics.MeanMetric().to(device)
    total_metric = torchmetrics.MeanMetric().to(device)

    if args.amp_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    for _, data in enumerate(loader):
        if args.enable_dali:
            content = data[0]['content']
        else:
            content = data
            content = content.to(device)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.enable_amp):
            content_out, content_features, content_features_loss = model(content)
            content_loss, style_loss = loss_func(content, content_out, content_features, content_features_loss)
            total_loss = content_loss + style_loss

        # https://discuss.pytorch.org/t/whats-the-correct-way-of-using-amp-with-multiple-losses/93328/3
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        content_metric(content_loss)
        style_metric(style_loss)
        total_metric(total_loss)

    return content_metric.compute(), style_metric.compute(), total_metric.compute()

def val_style(model, loss_func, loader, device, args):

    model.eval()

    content_metric = torchmetrics.MeanMetric().to(device)
    style_metric = torchmetrics.MeanMetric().to(device)
    total_metric = torchmetrics.MeanMetric().to(device)
    
    if args.amp_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    for _, data in enumerate(loader):
        if args.enable_dali:
            content = data[0]['content']
        else:
            content = data
            content = content.to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=args.enable_amp):
            content_out, content_features, content_features_loss = model(content)
            content_loss, style_loss = loss_func(content, content_out, content_features, content_features_loss)
            total_loss = content_loss + style_loss

        content_metric(content_loss)
        style_metric(style_loss)
        total_metric(total_loss)

    return content_metric.compute(), style_metric.compute(), total_metric.compute()

def train_model(model, optimizer, loss, train_loader, val_loader, args):
                
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    checkpoint_count = 0
    init_epoch = 0
    loss.to(device)
    project_name = "StyleTransfer"
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    config = {
    "model": model.__class__.__name__,
    "optimizer": optimizer.__class__.__name__,
    "loss": loss.__class__.__name__,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "momentum": None if isinstance(optimizer, torch.optim.Adam) else optimizer.param_groups[0]['momentum'],
    }

    if args.resume_id is not None:
        model_, optimizer_, scaler_, epoch_, _, _ = utils.load_train_state("model/train.state")

        model.load_state_dict(model_)
        cmodel = torch.compile(model, mode="reduce-overhead")
        cmodel.to(device)

        optimizer.load_state_dict(optimizer_)
        scaler.load_state_dict(scaler_)

        init_epoch = epoch_ + 1 # PLus 1 means start at the next epoch
        run = wandb.init(project=project_name, config=config, id=args.resume_id, resume=True)
    else:
        cmodel = torch.compile(model, mode="reduce-overhead")
        cmodel.to(device)
        run = wandb.init(project=project_name, config=config)

    for epoch in tq.tqdm(range(init_epoch, args.epochs), total=args.epochs, desc='Epochs', initial=init_epoch):
        train_loss = train_style(cmodel, optimizer, scaler, loss, train_loader, device, args)
        val_loss = val_style(cmodel, loss, val_loader, device, args)
        wandb.log({"content_loss": train_loss[0], 
                    "style_loss": train_loss[1], 
                    "total_loss": train_loss[2], 
                    "content_loss_val": val_loss[0], 
                    "style_loss_val": val_loss[1], 
                    "total_loss_val": val_loss[2], 
                    "epoch": epoch})
        
        checkpoint_count = checkpoint_count + 1 
        if checkpoint_count == args.checkpoint_freq:
            utils.save_train_state(model, optimizer, scaler, epoch, "model/train.state")
            checkpoint_count = 0
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Saved checkpoint at epoch: {epoch + 1} ({now})")

    model_scripted = torch.jit.trace(model.cpu(), (torch.rand(1,3,256,256),torch.rand(1,3,256,256)))
    model_scripted.save('model/model_style.pt')
    run.finish()

def train_model_no_wandb(model, optimizer, loss, train_loader, val_loader, args):
                
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    checkpoint_count = 0
    init_epoch = 0
    loss.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp)

    train_list = []
    val_list = []

    if args.resume_id is not None:
        model_, optimizer_, scaler_, epoch_, train_list_, val_list_ = utils.load_train_state("model/train.state")

        model.load_state_dict(model_)
        cmodel = torch.compile(model, mode="reduce-overhead")
        cmodel.to(device)

        train_list = train_list_
        val_list = val_list_

        optimizer.load_state_dict(optimizer_)
        scaler.load_state_dict(scaler_)

        init_epoch = epoch_ + 1 # PLus 1 means start at the next epoch
    else:
        cmodel = torch.compile(model, mode="reduce-overhead")
        cmodel.to(device)

    for epoch in tq.tqdm(range(init_epoch, args.epochs), total=args.epochs, desc='Epochs', initial=init_epoch):
        train_loss = train_style(cmodel, optimizer, scaler, loss, train_loader, device, args)
        train_list.append(train_loss[2].cpu())

        val_loss = val_style(cmodel, loss, val_loader, device, args)
        val_list.append(val_loss[2].cpu())
        
        checkpoint_count = checkpoint_count + 1
        if checkpoint_count == args.checkpoint_freq:
            utils.save_train_state(model, optimizer, scaler, epoch, "model/train.state", train_list, val_list)
            checkpoint_count = 0
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Saved checkpoint at epoch: {epoch + 1} ({now})")

    model_scripted = torch.jit.trace(model.cpu(), (torch.rand(1,3,256,256),torch.rand(1,3,256,256)))
    model_scripted.save('model/model_style.pt')
