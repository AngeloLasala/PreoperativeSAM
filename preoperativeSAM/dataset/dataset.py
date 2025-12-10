"""
Dataset and data augmentation for 2D medical image segmentation.
"""

import os
from random import randint
import numpy as np
import torch
from skimage import color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from typing import Callable
import os
import cv2
import pandas as pd
import random
import json
import matplotlib.pyplot as plt

def to_long_tensor(pic):
    """
    Convert a PIL image or NumPy array into a PyTorch LongTensor.

    This function ensures that the input label or mask is converted 
    into a tensor of type torch.int64 (LongTensor), which is required 
    by PyTorch loss functions such as CrossEntropyLoss or NLLLoss 
    that expect class indices as integer labels (not one-hot or float tensors).

    Parameters:
    ----------
        pic (PIL.Image or numpy.ndarray): Input image or mask. 
            Typically a 2D array where each pixel represents a class index.

    Returns:
    --------
        torch.LongTensor: Tensor containing the same data as `pic`, 
        converted to integer (int64) type.
    """
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def correct_dims(*images):
    """
    Function that uniforms the dimention of input images.
    Grey: (H, W)    -> (H, W, 1)
    RGB:  (H, W, 3) -> (H, W, 3) 

    Parameters:
    -----------
        images: multiple input of images

    Returns:
    -------
        corr_images: list of corrected images
    """
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def random_click(mask, class_id=1):
    """
    Generate a random click point on a segmentation mask.

    This function simulates a user click on a segmentation mask for 
    interactive segmentation tasks. It randomly selects a pixel 
    coordinate belonging to the specified class (`class_id`). 
    If no pixel of that class is present in the mask, it instead 
    selects a pixel from the background (any other class).

    Parameters:
    -----------
        mask (numpy.ndarray): 2D array where each pixel value 
            represents a class index.
        class_id (int, optional): Class ID to click on. 
            Defaults to 1.

    Returns:
    --------
        tuple:
            - pt (numpy.ndarray): Array of shape (1, 2) containing 
              the [x, y] coordinates of the randomly selected point.
            - point_label (list of int): [1] if the click is on the 
              target class (positive click), or [0] if it's on another 
              class (negative click).
    """

    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]

def fixed_click(mask, class_id=1):
    """
    Fixed click w/o randomicity
    """
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[len(indices)//2] 
    return pt[np.newaxis, :], [point_label]


def random_clicks(mask, class_id = 1, prompts_number=10):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt_index = np.random.randint(len(indices), size=prompts_number)
    pt = indices[pt_index]
    point_label = np.repeat(point_label, prompts_number)
    return pt, point_label

def pos_neg_clicks(mask, class_id=1, pos_prompt_number=5, neg_prompt_number=5):
    pos_indices = np.argwhere(mask == class_id)
    pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
    pos_prompt_indices = np.random.randint(len(pos_indices), size=pos_prompt_number)
    pos_prompt = pos_indices[pos_prompt_indices]
    pos_label = np.repeat(1, pos_prompt_number)

    neg_indices = np.argwhere(mask != class_id)
    neg_indices[:, [0,1]] = neg_indices[:, [1,0]]
    neg_prompt_indices = np.random.randint(len(neg_indices), size=neg_prompt_number)
    neg_prompt = neg_indices[neg_prompt_indices]
    neg_label = np.repeat(0, neg_prompt_number)

    pt = np.vstack((pos_prompt, neg_prompt))
    point_label = np.hstack((pos_label, neg_label))
    return pt, point_label

def random_bbox(mask, class_id=1, img_size=256):
    # return box = np.array([x1, y1, x2, y2])
    indices = np.argwhere(mask == class_id) # Y X
    indices[:, [0,1]] = indices[:, [1,0]] # x, y
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    classw_size = maxx-minx+1
    classh_size = maxy-miny+1

    shiftw = randint(int(0.95*classw_size), int(1.05*classw_size))
    shifth = randint(int(0.95*classh_size), int(1.05*classh_size))
    shiftx = randint(-int(0.05*classw_size), int(0.05*classw_size))
    shifty = randint(-int(0.05*classh_size), int(0.05*classh_size))

    new_centerx = (minx + maxx)//2 + shiftx
    new_centery = (miny + maxy)//2 + shifty

    minx = np.max([new_centerx-shiftw//2, 0])
    maxx = np.min([new_centerx+shiftw//2, img_size-1])
    miny = np.max([new_centery-shifth//2, 0])
    maxy = np.min([new_centery+shifth//2, img_size-1])

    return np.array([minx, miny, maxx, maxy])

def fixed_bbox(mask, class_id = 1, img_size=256):
    indices = np.argwhere(mask == class_id) # Y X (0, 1)
    indices[:, [0,1]] = indices[:, [1,0]]
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])

class JointTransform2D:
    """
    Data augmentation on image and mask. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, img_size=256, low_img_size=256, ori_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):

        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)

        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)

        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))

        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)

        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)

        # color transforms,  ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
    
        image = F.to_tensor(image)


        if not self.long_mask:
            mask = F.to_tensor(mask)
            low_mask = F.to_tensor(low_mask)
        else:
            mask = to_long_tensor(mask)
            low_mask = to_long_tensor(low_mask)

        return image, mask, low_mask


class IntroperativeiUS(Dataset):
    """
    Dataset class for Intraoprerative iUS segmentation. The structure of the folder is compatible with 
    future direction of the project, i.e., using preopreative information for guiding segmentation. 
    In line with this goal the dataset is structured in 'subjects' folders

    Folder structure:
        main_path
        ├── dataset_name
        |    ├── pre 
        |    |    ├── subject_name
        |    |    |    ├── img
        |    |    |    |    ├── subject_name_img_001.png
        |    |    |    |    ├── subject_name_img_002.png
        |    |    |    ├── label
        |    |    |    |    ├── subject_name_label_001.png
        |    |    |    |    ├── subject_name_label_002.png
        |    ├── post
        |    |    ├── subject_name
        |    |    |    ├── img
        |    |    |    |    ├── subject_name_img_001.png
        |    |    |    |    ├── subject_name_img_002.png
        |    |    |    ├── label
        |    |    |    |    ├── subject_name_label_001.png
        |    |    |    |    ├── subject_name_label_002.png

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self,
                main_path: str,
                dataset_name: str, 
                split: str, 
                joint_transform: Callable = None, 
                img_size = 256, 
                prompt = "click", 
                class_id = 1,
                one_hot_mask: int = False) -> None:

        # dataset path
        self.main_path = main_path
        self.dataset_name = dataset_name
        self.split = split
    
        self.one_hot_mask = one_hot_mask
        
        self.data_list = self.get_data_list()

        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        # self.class_dict_file = os.path.join(dataset_path, 'MainPatient/class.json')
        # with open(self.class_dict_file, 'r') as load_f:
        #     self.class_dict = json.load(load_f)
        if joint_transform is not None:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        class_id0, image, mask, filename = self.get_image_label(idx)
        
        ## correct dimensions if needed
        image, mask = correct_dims(image, mask)  

        ## data augmentation on the fly, TO UPDATE ...
        image, mask, low_mask = self.joint_transform(image, mask)
        # image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        # --------- make the point prompt -----------------
        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                #class_id = randint(1, classes-1)
                class_id = int(class_id0)
            elif 'val' in self.split:
                class_id = int(class_id0)
            else:
                class_id = self.class_id

            if 'train' in self.split:
                pt, point_label = random_click(np.asarray(mask), class_id)
                bbox = random_bbox(np.asarray(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.asarray(mask), class_id)
                bbox = fixed_bbox(np.asarray(mask), class_id, self.img_size)

            mask[mask!=class_id] = 0
            mask[mask==class_id] = 1
            low_mask[low_mask!=class_id] = 0
            low_mask[low_mask==class_id] = 1
            point_labels = np.array(point_label)
       
        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask':low_mask,
            'image_name': filename,
            'class_id': class_id,
            }


    def get_image_label(self, idx):
        """
        Get image and label
        """
        image_info = self.data_list[idx]

        img_path = os.path.join(self.main_path, self.dataset_name, image_info[0], image_info[1].split('_')[0], 'img', image_info[1])
        mask_path = os.path.join(self.main_path, self.dataset_name, image_info[0], image_info[1].split('_')[0], 'label', image_info[1])
        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        return self.class_id, image, mask, image_info[1]
        

    def get_data_list(self):
        """
        From patients name get the data list
        Note: this function take both pre and post for unified training
        subject_i -> pre/subject_i and post/subject_i
        """
        json_path = os.path.join(self.main_path, self.dataset_name, 'splitting.json')
        with open(json_path, 'r') as f:
            splitting_dict = json.load(f)
      

        subject_list = splitting_dict[self.split]
        data_list = []
        for subject in subject_list:
            pre_path = os.path.join(self.main_path, self.dataset_name, 'pre', subject)
            for img_name in os.listdir(os.path.join(pre_path,'img')):
                data_pre = ['pre', img_name]
                data_list.append(data_pre)
            
            post_path = os.path.join(self.main_path, self.dataset_name, 'post', subject)
            for img_name in os.listdir(os.path.join(post_path,'img')):
                data_post = ['post', img_name]
                data_list.append(data_post)

        return data_list

class PrePostiUS(Dataset):
    """
    Dataset class for Intraoprerative iUS segmentation. The structure of the folder is compatible with 
    future direction of the project, i.e., using preopreative information for guiding segmentation. 
    In line with this goal the dataset is structured in 'subjects' folders

    Folder structure:
        main_path
        ├── dataset_name
        |    ├── pre 
        |    |    ├── subject_name
        |    |    |    ├── img
        |    |    |    |    ├── subject_name_img_001.png
        |    |    |    |    ├── subject_name_img_002.png
        |    |    |    ├── label
        |    |    |    |    ├── subject_name_label_001.png
        |    |    |    |    ├── subject_name_label_002.png
        |    ├── post
        |    |    ├── subject_name
        |    |    |    ├── img
        |    |    |    |    ├── subject_name_img_001.png
        |    |    |    |    ├── subject_name_img_002.png
        |    |    |    ├── label
        |    |    |    |    ├── subject_name_label_001.png
        |    |    |    |    ├── subject_name_label_002.png

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self,
                main_path: str,
                dataset_name: str, 
                split: str, 
                joint_transform: Callable = None, 
                img_size = 256, 
                prompt = "click", 
                degree_prompt = 0,
                class_id = 1,
                one_hot_mask: int = False) -> None:

        # dataset path
        self.main_path = main_path
        self.dataset_name = dataset_name
        self.split = split
    
        self.one_hot_mask = one_hot_mask
        
        self.data_list, self.subject_list = self.get_data_list()

        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.degree_prompt = degree_prompt    ## number of img/mask/text to use as prompt information
        
        if joint_transform is not None:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        class_id0, image, mask, filename, img_prompt, mask_prompt, text_prompt = self.get_image_label(idx)

        ## correct dimensions if needed & Data Augumentation
        image, mask = correct_dims(image, mask)  
        image, mask, low_mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        # --------- make the point prompt -----------------
        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                #class_id = randint(1, classes-1)
                class_id = int(class_id0)
            elif 'val' in self.split:
                class_id = int(class_id0)
            else:
                class_id = self.class_id

            if 'train' in self.split:
                pt, point_label = random_click(np.asarray(mask), class_id)
                bbox = random_bbox(np.asarray(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.asarray(mask), class_id)
                bbox = fixed_bbox(np.asarray(mask), class_id, self.img_size)

            mask[mask!=class_id] = 0
            mask[mask==class_id] = 1
            low_mask[low_mask!=class_id] = 0
            low_mask[low_mask==class_id] = 1
            point_labels = np.array(point_label)
       
        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        if img_prompt != 0 and mask_prompt != 0 and text_prompt != 0:

            ## Processinf prompt info
            imgs_p, masks_p, low_masks_p = [], [], []
            for img_p, mask_p in zip(img_prompt, mask_prompt):
                img_p, mask_p = correct_dims(img_p, mask_p)
                imag_p, mask_p, low_mask_p = self.joint_transform(img_p, mask_p)
                
                imgs_p.append(imag_p[0,:,:])
                masks_p.append(mask_p)
                low_masks_p.append(low_mask_p)

            img_prompt = torch.stack(imgs_p, dim=0)
            mask_prompt = torch.stack(masks_p, dim=0)

        return {
                'image': image,
                'label': mask,
                'p_label': point_labels,
                'pt': pt,
                'bbox': bbox,
                'low_mask':low_mask,
                'image_name': filename,
                'class_id': class_id,
                'img_prompt': img_prompt,
                'mask_prompt': mask_prompt,
                'text_prompt': text_prompt
                }
            
    def get_image_label(self, idx):
        """
        Get image label and multimodal prompt
        """
        image_info = self.data_list[idx]
        
        ## Post iUS - input iUS and target mask
        img_path = os.path.join(self.main_path, self.dataset_name, image_info[0], image_info[1].split('_')[0], 'img', image_info[1])
        mask_path = os.path.join(self.main_path, self.dataset_name, image_info[0], image_info[1].split('_')[0], 'label', image_info[1])
        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        if self.degree_prompt > 0: ## Pre iUS - prompt information
            subject_prompt_dir = os.path.join(self.main_path, self.dataset_name, 'pre', image_info[1].split('_')[0])
            img_prompt_path = os.path.join(subject_prompt_dir, 'img')
            prompt_images_list = [f for f in os.listdir(img_prompt_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Seleziona 3 immagini casuali (o tutte se <3)
            num_samples = min(self.degree_prompt, len(prompt_images_list))
            selected_prompts = random.sample(prompt_images_list, num_samples)

            prompt_imgs, prompt_masks = [], []
            for i in selected_prompts:
                img_prompt = os.path.join(subject_prompt_dir, 'img', i)
                mask_prompt = os.path.join(subject_prompt_dir, 'label', i)
                text_prompt = os.path.join(subject_prompt_dir, 'text', 'text_prompt.txt')
                
                img_prompt = cv2.imread(img_prompt, 0)
                mask_prompt = cv2.imread(mask_prompt,0)

                prompt_imgs.append(img_prompt)
                prompt_masks.append(mask_prompt)

            # Leggi il testo del prompt
            text_prompt_path = os.path.join(subject_prompt_dir, 'text', 'text_prompt.txt')
            if os.path.exists(text_prompt_path):
                with open(text_prompt_path, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
            else:
                text_prompt = ""

            return self.class_id, image, mask, image_info[1], prompt_imgs, prompt_masks, text_prompt
        
        else: # No preoperative information
            return self.class_id, image, mask, image_info[1], 0, 0, 0
        

    def get_data_list(self):
        """
        From patients name get the data list
        subject_i -> post/subject_i
        """
        json_path = os.path.join(self.main_path, self.dataset_name, 'splitting.json')
        with open(json_path, 'r') as f:
            splitting_dict = json.load(f)
      
        subject_list = splitting_dict[self.split]
        
        data_list = []
        for subject in subject_list:
            post_path = os.path.join(self.main_path, self.dataset_name, 'post', subject)
            for img_name in os.listdir(os.path.join(post_path,'img')):
                data_post = ['post', img_name]
                data_list.append(data_post)

        return data_list, subject_list  

class PreIntraEndoscopy(Dataset):
    """
    Dataset class for Endoscopy (HENANCE) segmentation. The structure of the folder is compatible with 
    future direction of the project, i.e., using preopreative information for guiding segmentation. 
    In line with this goal the dataset is structured in 'subjects' folders

    Folder structure:
        main_path
        ├── dataset_name
        |    ├── pre 
        |    |    ├── subject_idx
        |    |    |    ├── img
        |    |    |    |    ├── subject_idx_info_img.png
        |    |    |    |    ├── ...
        |    |    |    ├── label
        |    |    |    |    ├── subject_idx_info_img.png
        |    |    |    |    ├── ...
        |    ├── intra
        |    |    ├── subject_idx
        |    |    |    ├── img
        |    |    |    |    ├── subject_idx_info_img.png
        |    |    |    |    ├── ...
        |    |    |    ├── label
        |    |    |    |    ├── subject_idx_info_img.png
        |    |    |    |    ├── ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """
    def __init__(self,
                main_path: str,
                dataset_name: str,
                pre_plus_intra: bool, 
                split: str, 
                joint_transform: Callable = None, 
                img_size = 256, 
                prompt = "click", 
                degree_prompt = 0,
                class_id = 1,
                one_hot_mask: int = False) -> None:

        ## Dataset path
        self.main_path = main_path
        self.dataset_name = dataset_name
        self.split = split

        # I want to distinguish between 'intra' only dataset and 'pre' + 'intra' dataset
        # pre_plus_intra = True -> dataset where I do not use preop info ad prompt BUT as input
        # pre_plus_intra = False -> dataset where I intra data as input AND eventually pre data as prompt 
        self.pre_plus_intra = pre_plus_intra  ## if True, data only from intraoper
        self.one_hot_mask = one_hot_mask
        
        self.data_list, self.subject_list = self.get_data_list()

        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.degree_prompt = degree_prompt    ## number of img/mask/text to use as prompt information
        
        if joint_transform is not None:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        pass

    def get_image_label(self, idx):
        """
        Get image label depend on configuration of trainin, i.e. pre_plus_intra or only intra
        """
        image_info = self.data_list[idx]
        
        ## Post iUS - input iUS and target mask
        img_path = os.path.join(self.main_path, self.dataset_name, image_info[0], image_info[1].split('_')[0], 'img', image_info[1])
        mask_path = os.path.join(self.main_path, self.dataset_name, image_info[0], image_info[1].split('_')[0], 'label', image_info[1])
        mask_path = mask_path.replace('.jpg', '.png')

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.degree_prompt > 0: ## Pre Endoscopic, I set numer of preop to use as prompt information
            print("Using preoperative information as prompt")
            subject_prompt_dir = os.path.join(self.main_path, self.dataset_name, 'pre', image_info[1].split('_')[0])
            img_prompt_path = os.path.join(subject_prompt_dir, 'img')
            prompt_images_list = [f for f in os.listdir(img_prompt_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(prompt_images_list) == 0: # check if intra subject has preoperative info
                # similar to no preopreative info
                return self.class_id, image, mask, image_info[1], 0, 0, 0

            # select random 'degree_prompt' images (or all if less than that)
            num_samples = min(self.degree_prompt, len(prompt_images_list))
            selected_prompts = random.sample(prompt_images_list, num_samples)
            
            prompt_imgs, prompt_masks = [], []
            for i in selected_prompts:
                img_prompt = os.path.join(subject_prompt_dir, 'img', i)
                mask_prompt = os.path.join(subject_prompt_dir, 'label', i)
                text_prompt = os.path.join(subject_prompt_dir, 'text', 'text_prompt.txt')
                
                img_prompt = cv2.imread(img_prompt, cv2.IMREAD_COLOR)
                img_prompt = cv2.cvtColor(img_prompt, cv2.COLOR_BGR2RGB)
                mask_prompt = cv2.imread(mask_prompt,0)

                prompt_imgs.append(img_prompt)
                prompt_masks.append(mask_prompt)

            # Leggi il testo del prompt, TO DO...
            text_prompt_path = os.path.join(subject_prompt_dir, 'text', 'text_prompt.txt')
            if os.path.exists(text_prompt_path):
                with open(text_prompt_path, 'r', encoding='utf-8') as f:
                    text_prompt = f.read().strip()
            else:
                text_prompt = ""

            return self.class_id, image, mask, image_info[1], prompt_imgs, prompt_masks, text_prompt
        
        else: # No preoperative information
            print("No preoperative information used as prompt")
            return self.class_id, image, mask, image_info[1], 0, 0, 0


    def get_data_list(self):
        """
        From patients name get the data list.
        Modular functio to get both dataset for 'intra' training with 'pre' prompt
        and the combine tarining with 'pre' and 'intra' as input.

        1) subject_i -> intra/subject_i
        2) subject_i -> pre/subject_i and intra/subject_i
        """
        json_path = os.path.join(self.main_path, self.dataset_name, 'splitting.json')
        with open(json_path, 'r') as f:
            splitting_dict = json.load(f)
      
        subject_list = splitting_dict[self.split]
        
        if not self.pre_plus_intra:  ## intra only dataset with pre as prompt
            data_list = []
            for subject in subject_list:
                post_path = os.path.join(self.main_path, self.dataset_name, 'intra', subject)
                for img_name in os.listdir(os.path.join(post_path,'img')):
                    data_post = ['intra', img_name]
                    data_list.append(data_post)

            return data_list, subject_list

        else:   ## combined pre + intra dataset as input data
            data_list = []
            for subject in subject_list:
                pre_path = os.path.join(self.main_path, self.dataset_name, 'pre', subject)
                for img_name in os.listdir(os.path.join(pre_path,'img')):
                    data_pre = ['pre', img_name]
                    data_list.append(data_pre)
                
                post_path = os.path.join(self.main_path, self.dataset_name, 'intra', subject)
                for img_name in os.listdir(os.path.join(post_path,'img')):
                    data_post = ['intra', img_name]
                    data_list.append(data_post)
            return data_list, subject_list

if __name__ == '__main__':
    from preoperativeSAM.cfg import get_config
    import matplotlib.patches as patches

    ## HENANCE DATASET
    opt = get_config("PreIntraEndo")

    low_image_size = 128       ## the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS
    encoder_input_size = 256   ## the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS
    degree_prompt = 0          ## how many preop images to use as prompt information

    transform = JointTransform2D(img_size=encoder_input_size, low_img_size=low_image_size, ori_size=opt.img_size, crop=opt.crop, 
                                p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, 
                                color_jitter_params=None, long_mask=True)

    dataset = PreIntraEndoscopy(main_path = opt.main_path, 
                                dataset_name = opt.dataset_name, 
                                split = opt.test_split, 
                                pre_plus_intra = False,
                                joint_transform = transform, 
                                img_size = opt.img_size,
                                degree_prompt = degree_prompt,
                                prompt = "click",
                                class_id = 1)

    
    print(dataset.get_image_label(0))

    exit()


    ## Single acquisition dataset iUS
    opt = get_config("PreDura")

    low_image_size = 128       ## the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS
    encoder_input_size = 256   ## the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS

    transform = JointTransform2D(img_size=encoder_input_size, low_img_size=low_image_size, ori_size=opt.img_size, crop=opt.crop, 
                                p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, 
                                color_jitter_params=None, long_mask=True)  
    
    dataset = IntroperativeiUS(main_path = opt.main_path, 
                                dataset_name = opt.dataset_name, 
                                split = opt.train_split, 
                                joint_transform = transform, 
                                img_size = opt.img_size,
                                prompt = "click",
                                class_id = 1)

    print(len(dataset))
    
    # idx = np.random.randint(0,100)
    # for i in range(10):
    #     data = dataset[idx]
         
    #     ig, axes = plt.subplots(1, 3, figsize=(10, 5), num=i)
    #     axes[0].imshow(data['image'][0], cmap='gray')
    #     axes[0].set_title("Immagine")
    #     # axes[0].axis('off')

    #     axes[1].imshow(data['image'][0], cmap='gray')
    #     axes[1].imshow(data['label'][0], alpha=0.2, cmap='jet')
    #     ## click
    #     x, y = data["pt"][0] 
    #     axes[1].scatter(x, y, c='red', s=80, marker='x', label='Click') 
    #     x_min, y_min, x_max, y_max = data["bbox"]
    #     ## bbox
    #     rect2 = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
    #                           linewidth=2, edgecolor='red', facecolor='none', label='BBox')
    #     axes[1].add_patch(rect2)
    #     axes[1].set_title("Immagine + Maschera")
    #     # axes[1].axis('off')

    #     axes[2].imshow(data['low_mask'][0], cmap='gray')
    #     axes[2].set_title("Immagine")
    # plt.show()

    ## Pre-Post dataset
    opt = get_config("PrePostiUS")

    low_image_size = 128       ## the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS
    encoder_input_size = 256   ## the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS
    degree_prompt = 3

    dataset = PrePostiUS(main_path = opt.main_path, 
                        dataset_name = opt.dataset_name, 
                        split = opt.test_split, 
                        joint_transform = transform, 
                        img_size = opt.img_size,
                        degree_prompt = degree_prompt,
                        prompt = "click",
                        class_id = 1)

    print(f'N imgs: {len(dataset)}')

    idx = np.random.randint(0,2)
    for i in range(5):
        data = dataset[idx]
         
        fig, axes = plt.subplots(2, 3, figsize=(20, 10), num=data["image_name"] + ' ' + str(i), tight_layout=True)
        axes[0,0].imshow(data['image'][0], cmap='gray')
        axes[0,0].set_title("Post resection - Input", fontsize=20)
        axes[0,0].axis('off')

        axes[0,1].imshow(data['image'][0], cmap='gray')
        axes[0,1].imshow(data['label'][0], alpha=0.2, cmap='jet')
        ## click
        x, y = data["pt"][0] 
        axes[0,1].scatter(x, y, c='red', s=80, marker='x', label='Click') 
        x_min, y_min, x_max, y_max = data["bbox"]
        ## bbox
        rect2 = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              linewidth=2, edgecolor='red', facecolor='none', label='BBox')
        axes[0,1].add_patch(rect2)
        axes[0,1].set_title("Input + self prompt", fontsize=20)
        axes[0,1].axis('off')
        

        axes[0,2].imshow(data['low_mask'][0], cmap='gray')
        axes[0,2].set_title("Mask - Output", fontsize=20)
        axes[0,2].axis('off')

        if (data['img_prompt'] != 0) and (data['mask_prompt'] != 0) and (data["text_prompt"] != 0):
            axes[1,0].imshow(data['img_prompt'][0], cmap='gray')
            axes[1,0].imshow(data['mask_prompt'][0], alpha=0.2, cmap='jet')
            axes[1,0].set_title(data["text_prompt"], fontsize=20)
            axes[1,0].axis('off')

            axes[1,1].imshow(data['img_prompt'][1], cmap='gray')
            axes[1,1].imshow(data['mask_prompt'][1], alpha=0.2, cmap='jet')
            axes[1,1].set_title(data["text_prompt"], fontsize=20)
            axes[1,1].axis('off')

            axes[1,2].imshow(data['img_prompt'][2], cmap='gray')
            axes[1,2].imshow(data['mask_prompt'][2], alpha=0.2, cmap='jet')
            axes[1,2].set_title(data["text_prompt"], fontsize=20)
            axes[1,2].axis('off')

    plt.show()