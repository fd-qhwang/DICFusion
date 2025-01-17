import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import torch.nn as nn
from utils.utils import *
import cv2

def ycrcb_to_bgr(one):
    one = one.astype('float32')
    Y, Cr, Cb = cv2.split(one)
    B = (Cb - 0.5) * 1. / 0.564 + Y
    R = (Cr - 0.5) * 1. / 0.713 + Y
    G = 1. / 0.587 * (Y - 0.299 * R - 0.114 * B)
    return cv2.merge([B, G, R])

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

    # 限制值在[0,1]范围内
    #gray = gray.clamp(0.0, 1.0)

    return gray


@MODEL_REGISTRY.register()
class DICFusionModel(BaseModel):

    def __init__(self, opt):
        super(DICFusionModel, self).__init__(opt)
        torch.set_float32_matmul_precision('high')
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.clip_grad_norm_value = 0.1

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.net_g.train()

        # Define losses
        self.losses = ['metric', 'content', 'content_mask', 'edge', 'align', 'smooth', 'bce_boundary', 'bce_binary', 'ohem']
        for loss_name in self.losses:
            if train_opt.get(f'{loss_name}_opt'):
                loss_criterion = build_loss(train_opt[f'{loss_name}_opt']).to(self.device)
                setattr(self, f'cri_{loss_name}', loss_criterion)
            else:
                setattr(self, f'cri_{loss_name}', None)

        # Ensure at least one loss criterion is defined
        if not any(getattr(self, f'cri_{loss_name}') for loss_name in self.losses):
            raise ValueError('All loss criterions are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.VIS = data['VIS'].to(self.device)
        self.IR = data['IR'].to(self.device)
        # 使用if语句检查数据中是否有这些键，并进行赋值
        if 'CBCR' in data:
            self.CBCR = data['CBCR'].to(self.device)
        if 'BD' in data:
            self.BD = data['BD'].to(self.device)
        if 'BI' in data:
            self.BI = data['BI'].to(self.device)
        if 'MASK' in data:
            self.MASK = data['MASK'].to(self.device)
        if 'LABEL' in data:
            self.LABEL = data['LABEL'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.data_fusion, self.boundary_out, self.binary_out, self.semantic_pred = self.net_g(self.VIS, self.IR)
        # 判断VIS是否是三通道
        if self.VIS.shape[1] == 3:
            VIS_Y = RGB2GRAY(self.VIS)  # 转换到Y通道
        else:
            VIS_Y = self.VIS

        # Loss computation phase
        loss_conditions = [
            ("cri_metric", "l_metric_scd", self.IR, self.VIS, self.data_fusion),
            ("cri_content_mask", "l_content_masx1", self.IR,self.VIS, self.data_fusion, self.MASK),
            #("cri_align", "l_align_max", torch.max(self.IR,self.VIS), self.data_fusion),
            ("cri_align", "l_align_sum", self.IR, self.VIS, self.data_fusion),
            ("cri_edge", "l_edge_fusion", self.IR, self.VIS, self.data_fusion),
            #("cri_smooth", "l_smooth_max", self.data_fusion),    
            ("cri_bce_boundary", "l_boundary", self.boundary_out, self.BD),
            ("cri_bce_binary", "l_binary", self.binary_out, self.BI),
            ("cri_ohem", "l_semantic", self.semantic_pred, self.LABEL)
        ]

        l_total_phase2, loss_phase2_dict = self.compute_losses(loss_conditions)
        # Backpropagation and optimization
        l_total_phase2.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.clip_grad_norm_value, norm_type=2)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_phase2_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # Local padded variables
        #VIS_padded, IR_padded, pads = self.pad_to(self.VIS, self.IR, stride=16)
        # Use the padded tensors for the operations
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.data_fusion, _, _, self.semantic_pred = self.net_g_ema(self.VIS, self.IR)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.data_fusion, _, _, self.semantic_pred = self.net_g(self.VIS, self.IR)
                #self.data_fusion = self.net_g(VIS_padded, IR_padded)
            self.net_g.train()

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

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.data_fusion, _, _, self.semantic_pred = self.net_g_ema(VIS_padded, IR_padded)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.data_fusion, _, _, self.semantic_pred = self.net_g(VIS_padded, IR_padded)
                self.net_g.train()

        # 裁剪回原始尺寸
        self.data_fusion = self.unpad(self.data_fusion, VIS_pads)  # 裁剪输出
        self.semantic_pred = self.unpad(self.semantic_pred, VIS_pads)  # 裁剪语义预测

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics')
        #with_metrics = (self.opt['val'].get('metrics') is not None) and (current_iter > self.phase1_iter)
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        compute_miou = False  # Default to True if the field is not present
        seg_metric = SegmentationMetric(15, device=self.device)
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['IR_path'][0]))[0]
            self.feed_data(val_data)
            #self.test()
            self.test_pad()
            visuals = self.get_current_visuals()
            #VIS_img = tensor2img([visuals['VIS']]) # 3 h w
            # 判断VIS是否是三通道
            if self.VIS.shape[1] == 3:
                VIS_Y = RGB2GRAY(self.VIS)  # 转换到Y通道
            else:
                VIS_Y = self.VIS
            #VIS_img = tensor2img(self.VIS.detach().cpu()) # 1 h w
            VIS_img = tensor2img(VIS_Y.detach().cpu()) # 1 h w
            IR_img = tensor2img([visuals['IR']]) # 1 h w
            CBCR_img = visuals['CBCR']
            fusion_img = tensor2img([visuals['result']])
            fusion_img=(fusion_img-np.min(fusion_img))/(np.max(fusion_img)-np.min(fusion_img))
            fusion_img, VIS_img, IR_img = fusion_img * 255, VIS_img * 255, IR_img * 255
            metric_data['img_fusion'] = fusion_img.squeeze(0)
            metric_data['img_A'] = VIS_img.squeeze(0)
            metric_data['img_B'] = IR_img.squeeze(0)
            if 'LABEL' in val_data:
                compute_miou = True
                self.seg_result = torch.argmax(self.semantic_pred, dim=1, keepdim=True)
                seg_metric.addBatch(self.seg_result, self.LABEL, [255])
            # tentative for out of GPU memory
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                vi_filepath = val_data['VIS_path'][0]
                fuse_y = fusion_img.squeeze(0)
                img_vi  = cv2.imread(vi_filepath, flags=cv2.IMREAD_COLOR)
                # get cb and cr channels of the visible image
                vi_ycbcr = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
                vi_y  = vi_ycbcr[:, :, 0]
                vi_cb = vi_ycbcr[:, :, 1]
                vi_cr = vi_ycbcr[:, :, 2]
                # get BGR-fused image
                fused_ycbcr = np.stack([fuse_y, vi_cb, vi_cr], axis=2).astype(np.uint8)
                fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)
                color_fusion = ycrcb_to_bgr(cv2.merge([fusion_img.transpose(1, 2, 0), CBCR_img]))
                #cv2.imwrite(save_img_path, fused_bgr)
                imwrite(save_img_path, fused_bgr)
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        if compute_miou == True:
            self.mIoU = np.array(seg_metric.meanIntersectionOverUnion().item())
            self.Acc = np.array(seg_metric.pixelAccuracy().item())
            logger = get_root_logger()
            logger.info('mIou: {:.4f}, Acc: {:.4f}\n'.format(self.mIoU, self.Acc))
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            best_metric_value = self.best_metric_results[dataset_name][metric]["val"]
            best_metric_iter = self.best_metric_results[dataset_name][metric]["iter"]
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
            if (value==best_metric_value) and (self.is_train==True):
                print(f'Saving best %s models and training states.' % metric)
                self.save_best(metric, best_metric_iter)

        logger = get_root_logger()
        #logger.info('mIou: {:.4f}, Acc: {:.4f}\n'.format(self.mIoU, self.Acc))
        if not self.is_train:
            print(log_str) # 由于某些原因在test阶段需要print才行
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['VIS'] = (self.VIS.detach().cpu())
        out_dict['IR'] = self.IR.detach().cpu()
        # 使用hasattr函数检查类实例中是否有这些属性，并进行赋值
        if hasattr(self, 'CBCR'):
            out_dict['CBCR'] = self.CBCR.detach().cpu().squeeze(0).numpy()
        if hasattr(self, 'BD'):
            out_dict['BD'] = self.BD.detach().cpu()
        if hasattr(self, 'BI'):
            out_dict['BI'] = self.BI.detach().cpu()
        if hasattr(self, 'LABEL'):
            out_dict['LABEL'] = self.LABEL.detach().cpu()
        if hasattr(self, 'data_fusion'):
            out_dict['result'] = self.data_fusion.detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()

        return out_dict


    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
