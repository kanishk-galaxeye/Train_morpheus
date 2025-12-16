import torch
import torch.nn as nn 
import torch.optim as optim 
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from data_transform import offset_crop, conditional_resize_pad, PercentileNormalize, RayleighMatchTensor
import os 
from torch.utils.tensorboard import SummaryWriter

train_dir = "/media/sphere/744c0eb8-a6d5-469f-9d27-449a9eef9aa1/home/admin123/Kanishk/classification/class/train"
val_dir = "/media/sphere/744c0eb8-a6d5-469f-9d27-449a9eef9aa1/home/admin123/Kanishk/classification/class/validation"
check_point_dir = "/media/sphere/744c0eb8-a6d5-469f-9d27-449a9eef9aa1/home/admin123/Kanishk/classification/checkpoint"
checkpoint_name = "real_noray_PadResize"
batch_size = 32
num_class = 5
num_epochs = 10
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter("runs/experiment1")

# define transformation 

train_transform = transforms.Compose([
    transforms.Lambda(offset_crop),
    transforms.ToTensor(),
    # RayleighMatchTensor(sigma=1),
    transforms.Normalize([0.5,0.5,0.5], [0.25, 0.25,0.25])
])

val_transform = transforms.Compose([

    transforms.Lambda(conditional_resize_pad),
    transforms.ToTensor(),
    PercentileNormalize(),
    transforms.Normalize([0.5,0.5,0.5], [0.25, 0.25,0.25])
])

# val_transform = transforms.Compose([

#     transforms.Lambda(conditional_resize_pad),
#     transforms.ToTensor(),
#     PercentileNormalize(),
#     transforms.Normalize([0.5,0.5,0.5], [0.25, 0.25,0.25])
# ])


### Define dataset


train_dataset = ImageFolder(root=train_dir, transform=train_transform)
val_dataset = ImageFolder(root = val_dir, transform=val_transform)

# Define dataloader

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False )



## Defining the model

model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)


## changing classification head to custum classes 

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, num_class)

# tranfering model to preffered device

model.to(device)




# loss 
criterion = nn.CrossEntropyLoss()

# optimizer

optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# training loop 
best_val_acc = 0.0
checkpoint = os.path.join(check_point_dir, checkpoint_name)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    val_run_loss = 0.0

    for images, labels  in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # calculating cross entropy
        loss = criterion(outputs, labels)
        # backporpogation
        loss.backward()
        # update weights
        optimizer.step()

        # calculating total loss 

        running_loss += loss.item()* images.size(0) # avg. loss in numpy * batch_size
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = 100*correct/total
    train_loss = running_loss/len(train_loader.dataset)
    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", train_acc,epoch)


    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            v_loss = criterion(outputs, labels)
            val_run_loss += v_loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total+=labels.size(0)
            val_correct  += (predicted==labels).sum().item()
    val_loss = val_run_loss/len(val_loader.dataset)
    val_acc = 100*val_correct/val_total
    writer.add_scalar("Val/Loss", val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)

    print(f"Epoch {epoch}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train acc: {train_acc: .2f}%")
    print(f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc=val_acc
        torch.save(model.state_dict(), checkpoint)

print("training Complete. Best accuracy:", best_val_acc)





