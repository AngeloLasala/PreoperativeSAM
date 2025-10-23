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
from typing import Callable
import os
import cv2
import pandas as pd
import random
import json
import matplotlib.pyplot as plt

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    """
    Function that uniforms the dimention of input images
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
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
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
        # color transforms || ONLY ON IMAGE
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
    future direction of the project, i.e., usingi preopreative information for guiding segmentation. 
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
    
        # self.one_hot_mask = one_hot_mask
        
        self.data_list = self.get_data_list()

        # self.prompt = prompt
        # self.img_size = img_size
        self.class_id = class_id
        # self.class_dict_file = os.path.join(dataset_path, 'MainPatient/class.json')
        # with open(self.class_dict_file, 'r') as load_f:
        #     self.class_dict = json.load(load_f)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        class_id0, image, mask = self.get_image_label(idx)
        
        # correct dimensions if needed
        print(image.shape, mask.shape)
        image, mask = correct_dims(image, mask)  
        print(image.shape, mask.shape)

        if self.joint_transform:
            print('sono qui')
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
                pt, point_label = random_click(np.array(mask), class_id)
                bbox = random_bbox(np.array(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
                bbox = fixed_bbox(np.array(mask), class_id, self.img_size)
            mask[mask!=class_id] = 0
            mask[mask==class_id] = 1
            low_mask[low_mask!=class_id] = 0
            low_mask[low_mask==class_id] = 1
            point_labels = np.array(point_label)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask':low_mask,
            'image_name': filename + '.png',
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

        return self.class_id, image, mask
        
        





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

        

if __name__ == '__main__':
    from preoperativeSAM.cfg import get_config
    opt = get_config("PreDura")
    
    dataset = IntroperativeiUS(main_path=opt.main_path, 
                                dataset_name=opt.dataset_name, 
                                split=opt.train_split, 
                                # joint_transform=JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0.5, p_rota=0.5, p_scale=0.5, p_gaussn=0.5, p_contr=0.5, p_gama=0.5, p_distor=0.5),
                                img_size=opt.img_size,
                                prompt="click",
                                class_id=1,
                                one_hot_mask=opt.classes)
    for image, mask in dataset:
        if np.sum(mask) > 0 :
            print('ok')
        else: print('no mask')