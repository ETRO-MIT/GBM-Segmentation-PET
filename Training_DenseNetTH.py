###################################################
######################IMPORTS######################
###################################################

import os

from monai import transforms

from monai.metrics import MSEMetric
from monai.networks.nets import DenseNet121
from monai import data
from monai.utils import set_determinism

import torch
import argparse
import random

import pandas as pd
import numpy as np

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
lr = 1e-4
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

def get_data(data_root, subjects_list):
      thresholds = pd.read_csv(os.path.join(data_root, 'thresholds_suv.csv'))
      images = []
      labels = []
      for subject in subjects_list:
        f = os.path.join(data_root, subject)
        images.append(os.path.join(f,'fet_suv.nii.gz'))
        labels.append(thresholds.loc[thresholds['Subject'] == subject]['Threshold'])
      labels = torch.as_tensor(np.array(labels))
      return images, labels

###################################################
####################DATALOADER#####################
###################################################


# GET TRANSFORMS
train_transforms = transforms.Compose([
                        transforms.ScaleIntensityRange(a_min=0 , a_max=5, b_min=0, b_max=1,clip=True), 
                        transforms.EnsureChannelFirst(),
                        transforms.Orientation(axcodes="RAS"),
                        transforms.Spacing(pixdim=(1.0, 1.0, 1.0), mode="trilinear"),
                        transforms.Resize((128,128,128), mode="trilinear"),
                        transforms.RandFlip(prob=0.5, spatial_axis=0),
                        transforms.RandRotate(range_x=0.15, range_y=0.15, range_z=0.15, prob=0.5)
                        ])
                        
val_transforms = transforms.Compose([
                        transforms.ScaleIntensityRange(a_min=0 , a_max=5, b_min=0, b_max=1,clip=True), 
                        transforms.EnsureChannelFirst(),
                        transforms.Orientation(axcodes="RAS"),
                        transforms.Spacing(pixdim=(1.0, 1.0, 1.0), mode="trilinear"),
                        transforms.Resize((128,128,128), mode="trilinear")
                        ])

folds = {'F1':{'train':['CV2', 'CV3', 'CV4', 'CV5'], 'val':['CV1']}, 'F2':{'train':['CV1', 'CV3', 'CV4', 'CV5'], 'val':['CV2']}, 'F3':{'train':['CV1', 'CV2', 'CV4', 'CV5'], 'val':['CV3']}, 'F4':{'train':['CV1', 'CV2', 'CV3', 'CV5'], 'val':['CV4']}, 'F5':{'train':['CV1', 'CV2', 'CV3', 'CV4'], 'val':['CV5']}}

for fold in ['F1', 'F2', 'F3', 'F4', 'F5']:
    print(f"Starting fold {fold}...")

    subjects_train = []
    for cv in folds[fold]['train']:
        subjects_train += subjects[cv]
    subjects_val = subjects[folds[fold]['val'][0]]
    
    
    train_images, train_labels = get_data(data_root, subjects_train)
    train_ds = data.ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms, label_transform=None)
    train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
    
    val_images, val_labels = get_data(data_root, subjects_val)
    val_ds = data.ImageDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    
    
    
    
    ###################################################
    #################MODEL DEFINITION##################
    ###################################################
    
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, norm='INSTANCE').to(device)
    
    loss_function = torch.nn.L1Loss(reduction='mean') #or MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    mse_metric = MSEMetric(reduction="mean")
    mse_metric_batch = MSEMetric(reduction="mean_batch")
    
    
    
    ###################################################
    #####################TRAINING######################
    ###################################################
    
    val_interval = 1
    best_metric = +99999999999999
    best_metric_epoch = +99999999999999
    epoch_loss_values = []
    metric_values = []
    max_epochs = 1000
    
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
    
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        if (epoch + 1) % val_interval == 0:
            model.eval()
            
            df = pd.DataFrame(columns=['Subject', 'Threshold'])
    
            for val_data in val_loader:
    
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    idx = 0
                    row = {}
                    
                    val_outputs = model(val_images)
                    mse_metric(y_pred=val_outputs, y=val_labels)
                    mse_metric_batch(y_pred=val_outputs, y=val_labels)
                        
                row = {'Subject':subjects_val[idx], 'Threshold':val_outputs[0]}
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)        
                df.to_csv(os.path.join(results_root, f'output_PredThres_{fold}.csv'))
                
                idx += 1
    
            metric = mse_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = mse_metric_batch.aggregate()
            mse_metric.reset()
            mse_metric_batch.reset()
            
            torch.save(model.state_dict(), os.path.join(model_root, f"DenseNetTH_latest_{fold}.pth"))
    
            if metric < best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_root, f"DenseNetTH_best_{fold}.pth"))
                print("saved new best metric model")
    
            print(f"Current epoch: {epoch+1} current MSE: {metric:.4f} ")
    
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

