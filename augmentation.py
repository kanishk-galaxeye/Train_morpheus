import numpy as np 
from data_transform import sample_gaussian_in_range, clip , gamma_enh
import cv2
import torch 

#  Augmentation
def mig_aug_train(gamma_stretch=[0.4, 0.8], mul = [3,6],additive = [0.3, 1]):
    def augmentation(img):
        arr = img.numpy()
        arr = (arr-arr.min())/(arr.max()- arr.min())
        gst= sample_gaussian_in_range(np.array(gamma_stretch).mean(), 1.5, gamma_stretch[0], gamma_stretch[1])
        add = sample_gaussian_in_range(np.array(additive).mean(), 1.5, additive[0], additive[-1])
        mul_val = sample_gaussian_in_range(np.array(mul).mean(), 1.5, low= mul[0], high=mul[1])
        arr = clip(gamma_enh(arr,gst)+np.random.gamma(add, 1/20, arr.shape), min = 5, max = 99.25)
        if arr.ndim ==2:
            arr = np.stack([arr, arr, arr], axis= 0)
        img_out = torch.from_numpy(arr.astype(np.float32))
        return img_out

    return augmentation


def tu_aug_train(paraml =[0.7,1], g_enh=[0.4, 0.8], clutter=[1.2,3]):
    def apply_gamma(img):
        arr = img.numpy()
        arr /= arr.max()

        param = sample_gaussian_in_range((paraml[0]+paraml[1])/2, 1.5, low = paraml[0], high=paraml[-1])
        enh = sample_gaussian_in_range((g_enh[0]+g_enh[1])/2, 1.5,low= g_enh[0], high=g_enh[-1])
        cl = sample_gaussian_in_range((clutter[0]+clutter[1])/2, 1.5, clutter[0], clutter[-1])
        
        cl_denom = sample_gaussian_in_range(15,sigma=1.5,low = 10, high=20)
        arr = clip(gamma_enh(arr*np.random.gamma(param, 1/20),enh, arr.shape) + np.random.gamma(cl, 1/20, arr.shape), min = 0.1)
        if arr.ndim == 2:
            arr = np.stack([arr,arr, arr], axis = 0)
        img_out = torch.from_numpy(arr.astype(np.float32))
        return img_out
    return apply_gamma
def civil_aug_train(mgam1 = [0.5, 0.8], mgam2=[20, 70], cl = [0.3, 0.8], agamma1= [0.3, 0.8], agamma2=[15,30]):
    def augmentation(img):
        arr = img.numpy()
        arr = (arr-arr.min())/(arr.max()- arr.min())
        mgam1_val = sample_gaussian_in_range(np.array(mgam1).mean(), 1.5, mgam1[0], mgam1[-1])
        mgam2_val = sample_gaussian_in_range(np.array(mgam2).mean(),1.5, mgam2[0], mgam2[-1])
        agamma1_val = sample_gaussian_in_range(np.array(agamma1).mean(), 1.5, agamma1[0], agamma1[-1])
        agamma2_val = sample_gaussian_in_range(np.array(agamma2).mean(), 1.5, agamma2[0], agamma2[-1])
        cl_val = sample_gaussian_in_range(np.array(cl).mean(), 1.5, cl[0], cl[1])
        arr = clip(gamma_enh(arr*np.random.gamma(mgam1_val, 1/mgam2_val), cl_val) + np.random.gamma(agamma1_val, 1/agamma2_val,arr.shape),min = 10, max = 98)
        arr = arr
        if arr.ndim ==2:
            arr = np.stack([arr, arr, arr], axis = 0)
        img_out = torch.from_numpy(arr.astype(np.float32))
        return img_out
    return augmentation

#mgam1 = [3,6] agamma = [3,7]
def hel_aug_train(mgam1 = [0.6,2], mgam2=[15,20], agamma1= [3,7], agamma2=[40,70], cl= [0.3, 0.8]):
    def augmentation(img):
        arr = img.numpy()
        arr = (arr-arr.min())/(arr.max()- arr.min())
        mgam1_val = sample_gaussian_in_range(np.array(mgam1).mean(), 1.5, mgam1[0], mgam1[-1])
        mgam2_val = sample_gaussian_in_range(np.array(mgam2).mean(),1.5, mgam2[0], mgam2[-1])
        agamma1_val = sample_gaussian_in_range(np.array(agamma1).mean(), 1.5, agamma1[0], agamma1[-1])
        agamma2_val = sample_gaussian_in_range(np.array(agamma2).mean(), 1.5, agamma2[0], agamma2[-1])
        # cv2.resize(clip(gamma(sim* np.random.gamma(4,1/20,sim.shape),0.8) + np.random.gamma(.6, 1/20, sim.shape), min=1,max = 99.25),None, fx = 1, fy =1, interpolation =cv2.INTER_AREA)
        arr = clip(gamma_enh(arr*np.random.gamma(mgam1_val, 1/mgam2_val), 0.4) + np.random.gamma(agamma1_val, 1/agamma2_val,arr.shape),min = 0.5, max = 99.25)
        arr = arr
        if arr.ndim ==2:
            arr = np.stack([arr, arr, arr], axis = 0)
        img_out = torch.from_numpy(arr.astype(np.float32))
        return img_out
    return augmentation