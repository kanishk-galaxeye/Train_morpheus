import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from data_transform import conditional_resize_pad, offset_crop, PercentileNormalize, RayleighMatchTensor, LoadUint16TIff
import random
num_classes = 3
weights = "/media/sphere/744c0eb8-a6d5-469f-9d27-449a9eef9aa1/home/admin123/Kanishk/classification/checkpoint/sim_cls_83.pth"


def load_model(weights, num_classes, device = 'cuda'):
    model = models.mobilenet_v3_small(weights = None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    weights = torch.load(weights, map_location=device)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_type = 'real'):
    image = image_path
    transform = None
    if image_type == 'real':
        transform = transforms.Compose([
            LoadUint16TIff(gamma=1, minp=5, maxp=98.5, img_size=128), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(offset_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25])

        ])
    # ipdb.set_trace()
    return transform(image).unsqueeze(0)


def predict(model, image_tensor, device = "cuda", topk =num_classes):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim =1)
        top_probs , top_labels = torch.topk(probs, topk)
        return top_probs.cpu().numpy()[0], top_labels.cpu().numpy()[0]
    

def inference(weights, num_classes,image_path, device, topk, image_type):
    import os 
    # ipdb.set_trace()
    root_dir = os.path.dirname(os.path.dirname(image_path))
    true_labels = sorted(os.listdir(root_dir))
    # print('True labels', true_labels)
    model = load_model(weights, num_classes, device)
    image_tensor = preprocess_image(image_path, image_type)
    prob, label  = predict(model, image_tensor, device, topk)

    
    return true_labels[label[0]], os.path.dirname(image_path).split('/')[-1]

if __name__ == "__main__":
    import glob
    import ipdb
    import os
    import numpy as np
    num_classes = 3
    topk = 3
    device = "cuda"
    image_type = "real"
    weights = "checkpoint_FBH/65_fighter_bomber_helicopter_100_32_dropout0.35_weight_decay"
    root_dir = "bomber_figther_helicopter/val"
    subdir = os.listdir(root_dir)
    tresut = {}
    for sub in subdir:
        text = ""
        sub_path = os.path.join(root_dir, sub)
        image_paths = glob.glob(sub_path+'/*')
        correct = 0
        total = 0
        for  i in image_paths:
            # ipdb.set_trace()
            if os.path.splitext(i)[-1] == ".tif":
                total += 1
                prediction, true_label = inference(weights,num_classes,i, device, topk, image_type)
                if prediction == true_label:
                    correct += 1
                text += f"{os.path.basename(i)} \nTrue Label - {true_label}\nPrediction - {prediction}\n\n\n"
                
            else:
                pass
        text += f"correct predictions - {correct}\ntotal images - {total}\n accuracy {np.round(correct/total, 2)}"
        print('-------------------------------------------')
        print(f"          Results of class {sub}")
        print(f"Correct Prediction - {correct}")
        print(f"Total {total}")
        print(f"Accracy {np.round(correct/total, 2)}")
        print('--------------------------------------------')
        textfile = os.path.join(sub_path, "predictions.txt")
        with open(textfile, "w") as f:
            f.write(text)
    
    # for i in range(5):
    #     cdir  = glob.glob(root_dir+"/*")
    #     random_dir = random.sample(cdir, 1)[0]
    #     # ipdb.set_trace()
    #     image_path = random.sample(glob.glob(random_dir+'/*'), 1)[0]
    #     weights = "/media/sphere/744c0eb8-a6d5-469f-9d27-449a9eef9aa1/home/admin123/Kanishk/classification/checkpoint/sim_cls_83.pth"
    # image_path = "bomber_figther_helicopter/val/helicopter/ICEYE_X20_GRD_SLEDF_4049621_20240429T222841_27.tif"
    # inference(weights, num_classes, image_path, device,topk=1, image_type=image_type)