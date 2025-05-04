import logging
import torch
from os import path as osp

from data import build_dataloader, build_dataset
from models import build_model
from utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from utils.options import parse_options


def test_pipeline(root_path):
    # 解析配置选项
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # 创建目录和初始化日志
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    
    # 简化日志输出，不显示详细配置
    logger.info(f"Testing {opt['name']}...")

    # 创建测试数据集和数据加载器
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # 创建并加载模型
    model = build_model(opt)
    
    # 开始测试
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing dataset: {test_set_name}')
        # 执行验证
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        logger.info(f'Testing on {test_set_name} completed')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
