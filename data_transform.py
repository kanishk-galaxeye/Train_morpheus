import torch 
from torchvision import transforms
# from torchvision.datasets import ImageFolder
import numpy as np 
from PIL import Image
from torchvision.transforms import functional as F 
from torchvision.datasets import ImageFolder
from PIL import ImageOps
import tifffile
from PIL import Image
import cv2



## Data Transfrom 



class RayleighMatchTensor(object):

    """
    Pytorch Transform to apply Rayleigh Matching to a 3 channel tensor
    by treating all pixels as a single distribution
    
    """

    def __init__(self, sigma = 0.1):
        self.sigma = sigma

    def __call__(self, tensor):
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, but got {tensor.shape[0]}, channels")
        

        img_np = tensor.numpy().astype(np.float32)
        flat = img_np.ravel()
        sorted_vals = np.sort(flat)
        cdf_vals = np.linspace(0,1, len(sorted_vals), endpoint=False)
        quantiles = np.interp(flat, sorted_vals, cdf_vals)

        out = self.sigma*np.sqrt(-2*np.log(1-quantiles+1e-10))

        out = out.reshape(img_np.shape)
        return torch.from_numpy(out).float()
    

class PercentileNormalize(object):
    """
    Normalizes a tensor based on the min/max values found at specified percentiles.
    This is robust against outliers in the pixel data.
    
    The input tensor should be (C, H, W) and contain float values (e.g., post-ToTensor).
    """
    def __init__(self, p_min: float = 1.0, p_max: float = 99.0):
        """
        Args:
            p_min (float): The lower percentile to use for the minimum value.
            p_max (float): The upper percentile to use for the maximum value.
        """
        if not (0 <= p_min < p_max <= 100):
            raise ValueError("p_min and p_max must be between 0 and 100, and p_min < p_max.")
            
        self.p_min = p_min
        self.p_max = p_max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # 1. Convert C x H x W tensor to a flattened NumPy array
        # We perform calculations on NumPy because it has efficient percentile calculation.
        flat_np = tensor.numpy().ravel()
        
        # 2. Find the min and max values based on the specified percentiles
        # We calculate over the entire image (all channels/pixels)
        v_min = np.percentile(flat_np, self.p_min)
        v_max = np.percentile(flat_np, self.p_max)
        
        # Ensure v_min is strictly less than v_max to prevent division by zero
        if v_max <= v_min:
            # If the range is zero (e.g., a constant-color image), return zeros or handle as needed
            return torch.zeros_like(tensor)
            
        # 3. Apply normalization: Clip values outside the percentile range and scale to [0, 1]
        
        # Clamp (Clip) the tensor to the percentile boundaries
        tensor_clamped = torch.clamp(tensor, v_min, v_max)
        
        # Scale: (x - min) / (max - min)
        normalized_tensor = (tensor_clamped - v_min) / (v_max - v_min)
        
        return normalized_tensor.float()

    def __repr__(self):
        return (f"{self.__class__.__name__}(p_min={self.p_min}, p_max={self.p_max})")
    


# def gamma(img: Image.Image, param = 0.4, minp= 10, maxp = 99):
#     img = img**(1/param)

#     minp = np.percentile(img, minp)
#     maxp  = np.percentile(img, maxp)
#     img = np.clip(img, minp, maxp)
#     img = (img-img.min())/(img.max() - img.min())
#     return img




def gamma( param=0.4, minp=10, maxp=99):
    # Convert PIL image to numpy array
    def apply_gamma(img):
        arr = img.numpy()

        # normalize to 0..1 if needed
        arr /= arr.max() + 1e-8

        # gamma correction
        arr = arr ** (1 / param)

        # percentile clipping
        lo = np.percentile(arr, minp)
        hi = np.percentile(arr, maxp)
        arr = np.clip(arr, lo, hi)

        # scale 0..1
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        if arr.ndim == 2:
            arr = np.stack([arr,arr, arr], axis = 0)

        # convert back to PIL image if needed
        img_out = torch.from_numpy(arr)
        return img_out

    return apply_gamma




def offset_crop(img: Image.Image, crop_size = 128, offset_y = 10, offset_x = 0):
    """
    Crops a square region of size 'crop_size' centered around 
    (target_center_x, target_center_y) from a PIL Image.
    
    Args:
        img (PIL.Image): The input image.
        target_center_x (int): The X-coordinate (width index) of the desired center.
        target_center_y (int): The Y-coordinate (height index) of the desired center.
        crop_size (int): The dimension of the square crop (e.g., 20).
        
    Returns:
        PIL.Image: The cropped image.
    """
    def apply_offset(img:Image.Image):
    # Calculate the half-size of the crop
        centerx, centery = img.size[0]//2, img.size[1]//2
        half_size = crop_size//2
        
        # Calculate the top-left corner coordinates (y, x)
        # PIL/F.crop expects (top, left, height, width)
        
        # Left (x) coordinate: center_x - half_width
        left = centerx - half_size 
        
        # Top (y) coordinate: center_y - half_height
        top = centery - half_size -offset_y
        
        # Ensure coordinates are not negative (if image is too small)
        left = max(0, left)
        top = max(0, top)
        
        # Use F.crop(img, top, left, height, width)
        cropped_img = F.crop(img, top=top, left=left, height=crop_size, width=crop_size)
        
        return cropped_img
    return apply_offset


def conditional_resize_pad(img):
    """
    Applies conditional resizing or zero-padding to make the image 128x128.
    """
    
    img = img.convert('RGB')

    W, H = img.size
    TARGET_SIZE = 128
    # 1. Resizing (If Dimension > 128)
    if W > TARGET_SIZE or H > TARGET_SIZE:
        img = transforms.Resize((TARGET_SIZE, TARGET_SIZE))(img)
        W, H = img.size # Update dimensions

    # 2. Zero Padding (If Dimensions < 128)
    if W < TARGET_SIZE or H < TARGET_SIZE:
        pad_width = TARGET_SIZE - W
        pad_height = TARGET_SIZE - H
        
        # Calculate symmetric padding: (left, top, right, bottom)
        padding = (pad_width // 2, 
                   pad_height // 2, 
                   pad_width - pad_width // 2, 
                   pad_height - pad_height // 2)
        
        # Ensure image is in a mode that supports padding (e.g., convert P to RGB)
        if img.mode == 'P':
            img = img.convert('RGB')
            
        # Apply padding with fill=0 (zero padding, resulting in black)
        img = ImageOps.expand(img, border=padding, fill=0)

    return img



        


class LoadUint16TIff:
 
    def __init__(self, gamma , minp, maxp , img_size = 128):
        self.gamma = gamma 
        self.minp = minp
        self.maxp = maxp 
        self.img_size  = img_size

    def __call__(self, path):
        arr = tifffile.imread(path).astype(np.float32)
        arr = arr/arr.max()
        # np.array
        arr = arr**(1/self.gamma)
        # percentilecut

        minp = np.percentile(arr, self.minp)
        maxp = np.percentile(arr, self.maxp)
        image = np.clip(arr, minp, maxp)
        arr = (image - image.min())/(image.max() - image.min())

        H,W = arr.shape
        TARGET_SIZE = self.img_size
        # 1. Resizing (If Dimension > 128)
        arr = cv2.resize(arr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        # --------Originnal code ---------------------------   
        # if W > TARGET_SIZE or H > TARGET_SIZE:
        #     arr = cv2.resize(arr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        #     H,W = arr.shape # Update dimensions
         
        # # 2. Zero Padding (If Dimensions < 128)
        # if W < TARGET_SIZE or H < TARGET_SIZE:
        #     pad_width = TARGET_SIZE - W
        #     pad_height = TARGET_SIZE - H

            
        #     # Calculate symmetric padding: (left, top, right, bottom)
        #     padding = ((pad_height // 2, 
        #             pad_height - pad_height // 2), 
        #             (pad_width//2,pad_width - pad_width // 2))
            
        #     # Ensure image is in a mode that supports padding (e.g., convert P to RGB)
        #     arr = np.pad(arr, padding, mode='constant',constant_values=0)

        # -----------Original code above --------------------------------
        if arr.ndim ==2:
            arr = np.stack([arr, arr, arr], axis =0)

        else: 
            arr = arr.transpose(2,0,1)

        return torch.from_numpy(arr)
    



#------- Capella Augmentation --------------------------------


from scipy.stats import truncnorm
def clip(image1, min=0.25, max=99.5):
    min = np.percentile(image1, min)
    max = np.percentile(image1, max)
    image = np.clip(image1, min, max)
    image = (image - image.min())/(image.max() - image.min())
    return image 
def gamma_enh(image, param, clipin = True, min = 0.25, max = 99.5):
    image = image**(1/param)
    
    if clipin:
        image = clip(image, min, max)
    else:
        image = (image - image.min())/(image.max() - image.min())
    return image 
def sample_gaussian_in_range(mu, sigma, low, high, n=1):
    # Convert boundaries to standard normal units
    a = (low - mu) / sigma
    b = (high - mu) / sigma

    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n)



def capella_augmentation(image,  gamma_stretch = [0.5, 1.2], additive = [0.7, 2.2]):

    gst = sample_gaussian_in_range(np.array(gamma_stretch).mean(), 1.5, gamma_stretch[0], gamma_stretch[1])
    add = sample_gaussian_in_range(np.array(additive).mean(), 1.5, additive[0], additive[-1])
    return clip(gamma_enh(image, gst)+ np.random.gamma(add, 1/20, image.shape), min =1 ,max = 99.25)


def synspective_augmentation():
    pass 



# class CustomImageFolder(ImageFolder):
#     def __init__(self,root, class_transforms, default_transform=None):
#         super().__init__(root)
#         self.class_transforms = class_transforms
#         self.dfault_transform = default_transform or transforms.ToTensor()


#     def __getitem__(self,index):
#         path, label = self.samples[index]
#         image = Image.open(path).convert('RGB')

#         transform = self.class_transforms.get(label, self.defaul_transform)

from torch.utils.data import Dataset
import ipdb
class CustomDataset(Dataset):
    def __init__(self, root_dir,class_transforms):
        self.data = ImageFolder(root=root_dir)
        self.class_transform = class_transforms

        self.class_to_idx = self.data.class_to_idx
        self.labels_name = list(self.class_to_idx.keys())
        # pdb.set_trace()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data.samples[idx]
        image = Image.open(img_path).convert('RGB')
        key = self.labels_name[label]
        if key in self.class_transform:
            transform = self.class_transform[key]
            image = transform(image)
        else:
            print(f"Warning: NO specific transform found for class {key}. Applying default")
            image = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229, 0.224, 0.225])])
            

        return image, label


class_transform = {
    'TU_95': transforms.Compose([transforms.Lambda(offset_crop),
                                 transforms.ToTensor(),
                                 transforms.Lambda])
}


class customVal(Dataset):
    def __init__(self,root_dir, class_transform):
        self.data = ImageFolder(root = root_dir)
        self.class_transform = class_transform

        self.class_to_idx = self.data.class_to_idx
        self.labels_name = list(self.class_to_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data.samples[idx]
        image = img_path
        key = self.labels_name[label]
        if key in self.class_transform:
            transform = self.class_transform[key]
            image = transform(image)
        else:
            print(f"Warning: No specific transform for class {key}. Applying default")
            # image  = tifffile.imread(image)
            image = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        return image, label