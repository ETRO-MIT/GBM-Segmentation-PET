###################################################
######################IMPORTS######################
###################################################

import os
import time

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import Compose, MapTransform, Activations, AsDiscrete

from monai.metrics import DiceMetric, compute_surface_dice
from monai.networks.nets import DynUNet
from monai import data
from monai.data import decollate_batch
from monai.utils import set_determinism

from torch.optim.lr_scheduler import _LRScheduler

import torch
import argparse
import random

import numpy as np
import pandas as pd

###################################################
####################ENVIRONMENT####################
###################################################

set_determinism(seed=0)

num_CPU_cores = 16
num_workers_train = round(0.7*num_CPU_cores)
num_workers_val = num_CPU_cores - num_workers_train
num_init_workers_train = round(num_workers_train/2)
num_replace_workers_train = num_workers_train - num_init_workers_train
num_init_workers_val = round(num_workers_val/2)
num_replace_workers_val = num_workers_val - num_init_workers_val
VAL_AMP=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()



###################################################
####################PARAMETERS#####################
###################################################

size = (128,128,128)
batch_size = 2
max_epochs = 1000
val_interval = 10
lr = 1e-2
dropout_rate = 0.0

data_root = '/path/to/data/'
model_root = '/path/to/models/'


###################################################
#####################FUNCTIONS#####################
###################################################
random.seed(0)

subjects = {'CV1': [ ],           
            'CV2': [ ],
            'CV3': [ ],
            'CV4': [ ],
            'CV5': [ ],
            'test': [ ]}

def get_data(data_root, train_val_or_test):
  data_list = []
  for subject in subjects[train_val_or_test]:
    f = os.path.join(data_root, subject)
    dict = {}
    dict['image'] = os.path.join(f,'fet_suv.nii.gz')
    dict['label'] = os.path.join(f,'tum.nii.gz')
    dict['thres'] = os.path.join(f,'thres.nii.gz')
    data_list.append(dict)
  return data_list

class ConvertToBinaryLabelsd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = torch.where(d[key] == 1, 1, 0)
        return d

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


###################################################
####################DATALOADER#####################
###################################################

# GET TRANSFORMS
train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label", "thres"], ensure_channel_first=True),
    ConvertToBinaryLabelsd(keys="label"),
    transforms.NormalizeIntensityd(keys=["image"]),
    transforms.Orientationd(keys=["image", "label", "thres"], axcodes="RAS"),
    transforms.Spacingd(keys=["image", "label","thres"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest", "nearest")), 
    
    transforms.RandCropByPosNegLabeld(keys=["image", "label", "thres"], spatial_size=size, label_key="label", num_samples=2),
    transforms.RandZoomd(keys=["image", "label", "thres"], min_zoom=1, max_zoom=1.3, mode=("trilinear", "nearest", "nearest")),
    transforms.RandFlipd(keys=["image", "label", "thres"], prob=0.5, spatial_axis=0), #only allow lateral flipping (symmetrical axis)
    transforms.RandRotated(keys=["image", "label", "thres"], range_x=0.15, range_y=0.15, range_z=0.15, prob=0.5)
    ])
val_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label", "thres"], ensure_channel_first=True),
    ConvertToBinaryLabelsd(keys="label"),
    #transforms.ClipIntensityPercentilesd(keys=['image'], lower=0.5, upper=99.5),
    transforms.NormalizeIntensityd(keys=["image"]),
    transforms.Orientationd(keys=["image", "label", "thres"], axcodes="RAS"),
    transforms.Spacingd(keys=["image", "label", "thres"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest", "nearest"))])

    
folds = {'F1':{'train':['CV2', 'CV3', 'CV4', 'CV5'], 'val':['CV1']}, 'F2':{'train':['CV1', 'CV3', 'CV4', 'CV5'], 'val':['CV2']}, 'F3':{'train':['CV1', 'CV2', 'CV4', 'CV5'], 'val':['CV3']}, 'F4':{'train':['CV1', 'CV2', 'CV3', 'CV5'], 'val':['CV4']}, 'F5':{'train':['CV1', 'CV2', 'CV3', 'CV4'], 'val':['CV5']}}

for fold in ['F1', 'F2', 'F3', 'F4', 'F5']:
    print(f"Starting fold {fold}...")
    train_files = []
    for cv in folds[fold]['train']:
        train_files += get_data(data_root, cv)
    val_files = get_data(data_root, folds[fold]['val'][0])
    val_subjects = subjects[folds[fold]['val'][0]]
    
    # DATALOADER
    train_ds = data.Dataset(train_files, transform=train_transforms)
    val_ds = data.Dataset(val_files, transform=val_transforms)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers_train)
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers_val)
    
    df = pd.DataFrame(columns=['Subject', 'Dice', 'NSD', 'AVE'])
    
    ###################################################
    #################MODEL DEFINITION##################
    ###################################################
    
    sizes = size
    spacings = (1.0, 1.0, 1.0)
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    
    model = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        deep_supervision=True,
        deep_supr_num=2,
        norm_name='INSTANCE',
        dropout=dropout_rate
    ).to(device)
    
    loss_function = DiceCELoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    lr_scheduler = PolyLRScheduler(optimizer, lr, max_epochs)
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # DEFINE INFERENCE MODE
    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
        
        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
    
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    
    
    ###################################################
    #####################TRAINING######################
    ###################################################
    
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    
    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            
            img, labels, thres = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
                batch_data["thres"].to(device),
            )
            inputs = torch.cat((img, thres), dim=1)
    
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if len(outputs.size()) - len(labels.size()) == 1:
                    outputs = torch.unbind(outputs, dim=1)
                loss = sum(0.5**i * loss_function(p, labels) for i, p in enumerate(outputs))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                idx = 0
                row = {}

                for val_data in val_loader:
                    val_img, val_thres, val_labels = (
                        val_data["image"].to(device),
                        val_data["thres"].to(device),
                        val_data["label"].to(device),
                    )
                    val_inputs = torch.cat((val_img, val_thres), dim=1)
    
                    val_outputs = inference(val_inputs)
                    val_outputs = post_trans(val_outputs).to(device)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                    
                    dice = dice_metric(y_pred=val_outputs, y=val_labels)
                    nsd = compute_surface_dice(val_outputs, val_labels, class_thresholds=[1])
                    ave = abs(np.sum(val_outputs[0,0,:,:,:].detach().cpu().numpy()) - np.sum(val_labels[0,0,:,:,:].detach().cpu().numpy()))
                    
                    row['Subject'] = val_subjects[idx]
                    row['Dice'] = dice.cpu().item()
                    row['NSD'] = nsd.cpu().item()
                    row['AVE'] = ave
            
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    df.to_csv(f'results_SegFet_2channel_{fold}.csv')
        
                    idx += 1
                
                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                dice_metric.reset()
                dice_metric_batch.reset()
    
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_root, f"2C-UNet_best_{fold}.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                
                torch.save(
                    model.state_dict(),
                    os.path.join(model_root, f"2C-UNet_latest_{fold}.pth"),
                )
                
        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start


