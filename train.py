import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from  torchvision import transforms, models
from data_transform import offset_crop, LoadUint16TIff, customVal, CustomDataset, sample_gaussian_in_range
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
import cv2
from augmentation import *
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import yaml

# Define HyperParameters and Directory 
with open("HyperParameters.yaml") as f:
    pp = yaml.safe_load(f)

directories = pp['directories']
hyp = pp['HyperParameters']

train_dir = directories['train_dir']
val_dir = directories['val_dir']
checkpoint_dir = directories['checkpoint_dir']
batch_size = hyp['batch_size']
learning_rate = hyp['learing_rate']
num_classes = hyp['num_classes']
device = hyp['device']
checkpoint_name = directories['checkpoint_name']
num_epochs = hyp["num_epochs"]
writer  = SummaryWriter(f"path of logging")
train_class_transform = {"fighter": transforms.Compose([transforms.Lambda(lambda img: offset_crop(img)(img)),
                                                        transforms.ToTensor(),
                                                        transforms.Lambda(mig_aug_train()),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224,0.225])]),
                            "Tu_95": transforms.Compose([transforms.Lambda(lambda img: offset_crop(img, crop_size=95)(img)),
                                                         transforms.Lambda(tu_aug_train()),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224,0.225])]),
                            "helicopter": transforms.Compose([transforms.Lambda(lambda img: offset_crop(img, crop_size=85, offset_y=17)(img)),
                                                              transforms.ToTensor(),
                                                              transforms.Lambda(hel_aug_train()),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224,0.225])]),
                            "civilian": transforms.Compose([transforms.Lambda(lambda img: offset_crop(img,crop_size=85,offset_y=17)(img)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224,0.225])])                         
                         
                         
                         }

val_class_transform = {
                        "fighter": transforms.Compose([LoadUint16TIff(gamma=1, minp=2, maxp=98, img_size=128),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224,0.225])]),
                        "Tu_95": transforms.Compose([LoadUint16TIff(gamma=1, minp=2, maxp=98, img_size=128),
                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224,0.225])]),
                        "helicopter": transforms.Compose([LoadUint16TIff(gamma=1, minp=2, maxp=98, img_size=128),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224,0.225])])
}



def path_loader(path):
    return path

# Preparing the train and val loader 
train_dataset = CustomDataset(root_dir=train_dir, class_transforms=train_class_transform)
val_dataset = customVal(root_dir=val_dir, class_transform=val_class_transform)
train_loader = DataLoader(train_dataset, batch_size=hyp['train_batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyp['val_batch_size'], shuffle=False)


#Defining model 
model = models.mobilenet_v3_small(weights= models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features, num_classes)
model.classifier[-1] = nn.Dropout(p=hyp['dropout'], inplace=True)
for param in model.classifier.parameters():
    param.requires_grad = True
model.to(device)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
param_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(param_to_update, lr = learning_rate, weight_decay=hyp['weight_decay'])




# training loop 

class_names = ["TU", "Civilian", "Helicopter"]

best_val_acc = 0.0
checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
alpha = hyp['alpha']
w_acc = hyp['accuracy_weights']
w_loss = 1- w_acc
warmup_epochs = hyp['warmup_epochs'] 
best_epoch = 0
best_score = -10
patience = hyp['patience']
delta = hyp['delta']
epochs_no_imporve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    val_run_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*images.size(0)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    train_acc = 100*correct/total
    train_loss = running_loss/(len(train_loader.dataset))
    writer.add_scalar("Train/loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", train_acc, epoch)


    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    ema_acc = 0
    all_pred = []
    all_labels = []
    with torch.no_grad():
        for images , labels in val_loader:
            images, labels  = images.to(device), labels.to(device)
            outputs = model(images)
            v_loss = criterion(outputs, labels)
            val_run_loss += v_loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct = (predicted == labels).sum().item()
            all_pred.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_pred = np.concatenate(all_pred)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_pred)
    num_classes = cm.shape[0]
    total = cm.sum()

    val_loss = val_run_loss/len(val_loader.dataset)
    val_acc = 100*val_correct/val_total
    if epoch ==0:
        ema_acc == val_acc 
    score = w_acc*ema_acc - w_loss*loss 
    writer.add_scalar("Val/loss", val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)
    writer.add_scalar("val/ema_accuracy", ema_acc, epoch)
    writer.add_scalar("model/score", score, epoch)
    
    print(f"Epoch {epoch}/{num_epochs}")
    for i in range(num_classes):
        TP = cm[i,i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - (TP + FN + FP)

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        writer.add_scalar(f"TPR/{class_names[i]}", TPR*100, epoch)
        writer.add_scalar(f"FPR/{class_names[i]}", FPR*100, epoch)
        print(f"{class_names[i]}: TPR={TPR:.3f}, FPR={FPR:.3f}")
    print(f"Train Loss: {train_loss:.4f}, Train acc: {train_acc: .2f}%")
    print(f"Val Loss {val_loss:.2f} Val Acc: {val_acc:.2f}%")
    if epoch>= warmup_epochs:
        if score > best_score+ delta:
            best_score=score
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
            print("saving model at", best_epoch)
            epochs_no_improve = 0
        else:
            epochs_no_improve +=1
    if epochs_no_improve >= patience:
        print(f"Early Stop at epoch {epoch}")
        break
    

print("training Complete. epoch accuracy, loss :",best_epoch, best_val_acc, val_loss)
