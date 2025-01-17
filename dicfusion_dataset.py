from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import os
import torch
import numpy as np
import torch.utils.data as Data
import torch
# 加上这句话可以解决绝对引用的问题，但是同时导致了相对引用的问题
import sys,os
sys.path.append(os.getcwd())

from utils.registry import DATASET_REGISTRY
from utils.utils import randfilp
from natsort import natsorted
import cv2

def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def bgr_to_ycrcb(path):
    one = cv2.imread(path,1)
    one = one.astype('float32')
    (B, G, R) = cv2.split(one)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    return Y, cv2.merge([Cr,Cb])

@DATASET_REGISTRY.register()
class DICFusionDataset(Data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, opt):
        super(DICFusionDataset, self).__init__()
        self.opt = opt
        self.is_train = self.opt['is_train']
        self.img_size = int(self.opt['img_size']) if self.is_train else 128
        self.is_RGB = self.opt.get('is_RGB', False)  # VIS Default to GRAY

        # Use paths from the opts directly
        self.vis_folder = self.opt['dataroot_source2']
        self.ir_folder = self.opt['dataroot_source1']

        if self.is_train:
            self.bi_folder = self.opt['bi_folder_path']
            self.bd_folder = self.opt['bd_folder_path']
            self.label_folder = self.opt['dataroot_label']
            self.mask_folder = self.opt['mask_path']
        else:
            self.ir_folder = self.opt['dataroot_source1']
            self.vis_folder = self.opt['dataroot_source2']
            # Check if 'dataroot_label' exists in self.opt and it's not empty
            if 'dataroot_label' in self.opt and self.opt['dataroot_label'] != '~':
                self.label_folder = self.opt['dataroot_label']
            else:
                self.label_folder = None
            self.img_names = os.listdir(self.ir_folder)
            

        # Random crop size from opts
        self.crop = torchvision.transforms.RandomCrop(self.img_size)
        # gain infrared and visible images list
        self.ir_list = natsorted(os.listdir(self.ir_folder))
        print(len(self.ir_list))

    def __getitem__(self, index):
        # Gain image path
        image_name = self.ir_list[index]
        
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        


        # Read image as type Tensor
        vis = self._imread(path=vis_path, is_visible=True)
        ir = self._imread(path=ir_path, is_visible=False)

        if self.is_train:
            bi_path = os.path.join(self.bi_folder, image_name)
            bd_path = os.path.join(self.bd_folder, image_name)
            label_path = os.path.join(self.label_folder, image_name)
            
            mask_path = os.path.join(self.mask_folder, image_name)

            bi = self._imread(path=bi_path, label=True, is_visible=False)
            bd = self._imread(path=bd_path, label=True, is_visible=False)
            label = self._imread(path=label_path, label=True, is_visible=False)
            
            
            mask = self._imread(path=mask_path, label=False, is_visible=False)

            # Training images undergo certain data augmentations, including flip, rotation, and random cropping
            combined = torch.cat([vis, ir, bi, bd, label, mask], dim=1)
            if combined.shape[-1] <= self.img_size or combined.shape[-2] <= self.img_size:
                combined = TF.resize(combined, self.img_size)
            combined = randfilp(combined)
            combined = randfilp(combined)
            patch = self.crop(combined)

            vis, ir, bi, bd, label, mask = torch.split(patch, [3, 1, 1, 1, 1, 1], dim=1)
            vis, ir, bi, bd, label, mask = vis.squeeze(0), ir.squeeze(0), bi.squeeze(0), bd.squeeze(0), label.squeeze(0), mask.squeeze(0)
            label = label.type(torch.LongTensor)
            bi = bi / 255.0
            bd = bd / 255.0
            bi = bi.type(torch.LongTensor)
            bd = bd.type(torch.LongTensor)
            mask = mask.type(torch.LongTensor)
            if not self.is_RGB:
                vis = rgb2y(vis)
            data = {
                'VIS': vis,
                'IR': ir,
                'BI': bi,
                'BD': bd,
                'LABEL': label,
                'MASK': mask,
            }
            return data
        else:
            vis, ir = vis.squeeze(0), ir.squeeze(0)
            vis = rgb2y(vis)
            data = {
                'IR': ir,
                'VIS': vis,
                'IR_path': ir_path,
                'VIS_path': vis_path,
            }
            if self.is_RGB:
                _, cbcr = bgr_to_ycrcb(vis_path)
                data['CBCR'] = cbcr
            if self.label_folder:
                label_path = os.path.join(self.label_folder, image_name)
                label_tensor = self._imread(path=label_path, label=True)
                label_tensor = label_tensor.type(torch.LongTensor)
                label_tensor = label_tensor.squeeze(0)
                data['LABEL'] = label_tensor
                data['LABEL_path'] = label_path

                
            return data

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def _imread(path, label=False, is_visible=True):
        if path.endswith('.npy'):
            img = np.load(path)
            if img.ndim == 2:
                img = img[np.newaxis, np.newaxis, ...]  # 将 (H, W) 转换为 (1, 1, H, W)
            elif img.ndim == 3 and img.shape[2] in [1, 3]:
                img = img.transpose(2, 0, 1)[np.newaxis, ...]  # 将 (H, W, C) 转换为 (1, C, H, W)
            im_ts = torch.tensor(img, dtype=torch.float32)
        else:
            if label:
                img = Image.open(path)
                im_ts = TF.to_tensor(img).unsqueeze(0) * 255
            else:
                if is_visible:  # visible images; RGB channel
                    img = Image.open(path).convert('RGB')
                    im_ts = TF.to_tensor(img).unsqueeze(0)
                else:  # infrared images single channel
                    img = Image.open(path).convert('L')
                    im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts
