import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from archs import build_network
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import cv2
from utils.utils import *

def RGB2GRAY(rgb_image):
    """
    将RGB格式转换为GRAY格式。
    :param rgb_image: RGB格式的图像数据, 其shape为[B, C, H, W], 其中C=3。
    :return: 灰度图像
    """
    R = rgb_image[:, 0:1, :, :]
    G = rgb_image[:, 1:2, :, :]
    B = rgb_image[:, 2:3, :, :]

    # 使用上述公式计算灰度值
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray

@MODEL_REGISTRY.register()
class DICFusegModel(BaseModel):
    """图像融合和分割模型，仅用于测试"""

    def __init__(self, opt):
        super(DICFusegModel, self).__init__(opt)
        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        
        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        
        # 设置为评估模式
        self.net_g.eval()

    def feed_data(self, data):
        self.VIS = data['VIS'].to(self.device)
        self.IR = data['IR'].to(self.device)
        # 使用if语句检查数据中是否有这些键，并进行赋值
        if 'CBCR' in data:
            self.CBCR = data['CBCR'].to(self.device)
        if 'LABEL' in data:
            self.LABEL = data['LABEL'].to(self.device)

    def pad_to(self, tensor, padding_factor=8, pad_mode='replicate'):
        _, _, h, w = tensor.shape
        pad_h = (padding_factor - (h % padding_factor)) % padding_factor
        pad_w = (padding_factor - (w % padding_factor)) % padding_factor
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode=pad_mode), (pad_h, pad_w)

    def unpad(self, tensor, pads):
        pad_h, pad_w = pads
        return tensor[:, :, :tensor.shape[2] - pad_h, :tensor.shape[3] - pad_w]

    def test_pad(self):
        # 填充输入图像
        VIS_padded, VIS_pads = self.pad_to(self.VIS)
        IR_padded, IR_pads = self.pad_to(self.IR)

        with torch.no_grad():
            self.data_fusion, _, _, self.semantic_pred = self.net_g(VIS_padded, IR_padded)

        # 裁剪回原始尺寸
        self.data_fusion = self.unpad(self.data_fusion, VIS_pads)  # 裁剪输出
        self.semantic_pred = self.unpad(self.semantic_pred, VIS_pads)  # 裁剪语义预测

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics')
        
        # 创建进度条
        pbar = tqdm(total=len(dataloader), unit='image')
        
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['IR_path'][0]))[0]
            self.feed_data(val_data)
            self.test_pad()
            visuals = self.get_current_visuals()
            
            if self.VIS.shape[1] == 3:
                VIS_Y = RGB2GRAY(self.VIS)
            else:
                VIS_Y = self.VIS
            VIS_img = tensor2img(VIS_Y.detach().cpu())
            IR_img = tensor2img([visuals['IR']])
            CBCR_img = visuals['CBCR']
            fusion_img = tensor2img([visuals['result']])
            fusion_img=(fusion_img-np.min(fusion_img))/(np.max(fusion_img)-np.min(fusion_img))
            fusion_img, VIS_img, IR_img = fusion_img * 255, VIS_img * 255, IR_img * 255
            
            # 清理GPU内存
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['val']['suffix']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                vi_filepath = val_data['VIS_path'][0]
                fuse_y = fusion_img.squeeze(0)
                img_vi  = cv2.imread(vi_filepath, flags=cv2.IMREAD_COLOR)
                # 获取可见光图像的cb和cr通道
                vi_ycbcr = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
                vi_y  = vi_ycbcr[:, :, 0]
                vi_cb = vi_ycbcr[:, :, 1]
                vi_cr = vi_ycbcr[:, :, 2]
                # 获取BGR-融合图像
                fused_ycbcr = np.stack([fuse_y, vi_cb, vi_cr], axis=2).astype(np.uint8)
                fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)
                imwrite(save_img_path, fused_bgr)
                
            # 更新进度条
            pbar.update(1)
            pbar.set_description(f'Testing {img_name}')
            
        pbar.close()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['VIS'] = (self.VIS.detach().cpu())
        out_dict['IR'] = self.IR.detach().cpu()
        # 使用hasattr函数检查类实例中是否有这些属性，并进行赋值
        if hasattr(self, 'CBCR'):
            out_dict['CBCR'] = self.CBCR.detach().cpu().squeeze(0).numpy()
        if hasattr(self, 'LABEL'):
            out_dict['LABEL'] = self.LABEL.detach().cpu()
        if hasattr(self, 'data_fusion'):
            out_dict['result'] = self.data_fusion.detach().cpu()

        return out_dict
