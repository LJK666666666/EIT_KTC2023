"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
# os.environ['NCCL_P2P_DISABLE'] = "1"
# os.environ['NCCL_IB_DISABLE'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
# from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT
from diffusion import create_diffusion

from dataset import EITdataset
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from ema_pytorch import EMA

# 检测是否有 TPU 可用
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
    TPU_NUM_DEVICES = 8  # Kaggle TPU v5e-8
except ImportError:
    HAS_TPU = False
    TPU_NUM_DEVICES = 1

# 只在没有 TPU 的情况下导入 Accelerate
if not HAS_TPU:
    from accelerate import Accelerator
else:
    Accelerator = None


# from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def init_seed(seed=2019, reproducibility=True) -> None:
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def get_data_path(specified_path):
    """
    获取数据路径，如果指定路径不存在，则回退到默认路径'./data'

    Args:
        specified_path: 命令行参数指定的数据路径

    Returns:
        可用的数据路径
    """
    if specified_path and os.path.exists(specified_path):
        return specified_path
    return './data'


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args, index=0):
    """
    Trains a new DiT model.
    支持 TPU（单进程，自动使用所有 8 个核心）和 GPU 训练

    Args:
        args: 命令行参数
        index: 进程索引，在单进程模式下忽略
    """
    # 初始化设备和加速器
    if HAS_TPU:
        # TPU 模式 - 单进程，torch_xla 自动使用所有 8 个核心
        import torch_xla
        device = torch_xla.device()
        # 在单进程 TPU 模式下，rank = 0, world_size = 1
        # torch_xla 在后台自动处理所有 8 个核心的并行化
        rank = 0
        world_size = 1
        accelerator = None
        mixed_precision_mode = 'bf16'
        print(f"✅ TPU 设备: {device}（自动使用所有可用核心）")
    else:
        # GPU 模式，使用 Accelerator
        mixed_precision_mode = 'fp16'
        accelerator = Accelerator(mixed_precision=mixed_precision_mode)
        device = accelerator.device
        rank = 0
        world_size = 1
        print(f"✅ 使用 GPU 设备: {device}")

    seed = args.global_seed
    init_seed(seed)

    # 获取设备数量
    if HAS_TPU:
        # TPU 单进程模式：虽然有 8 个核心，但从训练脚本角度看是 1 个设备
        # torch_xla 会在内部自动将计算并行化到 8 个核心
        gpus = 1  # 单进程 = 单设备
    elif torch.cuda.is_available():
        gpus = torch.cuda.device_count()
    else:
        gpus = 1

    # 判断是否为主进程
    is_main_process = (rank == 0)

    # Setup an experiment folder:
    if is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = 'deit'  # args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        # logger = create_logger(checkpoint_dir)
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{checkpoint_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    model = DiT()

    #####################
    '''
    model_string_name = 'deit'  # args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(checkpoint_dir + '/best.pt', map_location='cpu')
    # #
    model.load_state_dict(state_dict["model"])'''

    # while 1 :pass
    model = model.to(device)

    # TPU：不直接转换模型，而是使用 autocast
    # 这样可以避免类型不匹配的问题
    #####################

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    # scheduler  = torch.optim.lr_scheduler.MultiStepLR(
    #     opt , milestones=[10000, 20000, 30000,40000], gamma=0.5
    # )
    # 获取设备名称
    if HAS_TPU:
        gpuname = "TPU"
    elif torch.cuda.is_available():
        gpuname = torch.cuda.get_device_name(0)
    else:
        gpuname = "CPU"
    modelname = 'DEIT'

    # 使用命令行参数或默认数据路径，如果指定路径不存在则回退到'./data'
    datapath = get_data_path(args.data_path)
     

    path = datapath + '/train/'
    dataset = EITdataset(path, modelname, backup_data_path='./data')

    path = datapath + '/valid/'
    dataVal = EITdataset(path, modelname, backup_data_path='./data')

    args.epochs = int(np.ceil(200000 / (len(dataset) / args.global_batch_size / gpus)))

    # 分布式训练：每个进程处理总批大小的 1/world_size
    # global_batch_size 是所有进程的总批大小
    # 每个进程的本地批大小 = global_batch_size / world_size
    if HAS_TPU and world_size > 1:
        batch_size = max(1, args.global_batch_size // world_size)
    else:
        batch_size = args.global_batch_size

    # TPU v5e-8 内存限制：15.75GB
    # 如果批大小过大，注意力机制会产生巨大的中间矩阵导致 OOM
    if HAS_TPU and batch_size > 4:
        logger.warning(f"⚠️  警告: TPU 批大小 {batch_size} 可能导致内存溢出！")
        logger.warning(f"    TPU v5e-8 只有 15.75GB 内存")
        logger.warning(f"    建议使用 --global-batch-size 2 或 4")

    logger.info(f"📊 实际使用的批大小: {batch_size}")

    # 为分布式训练创建采样器
    if HAS_TPU and world_size > 1:
        from torch.utils.data import DistributedSampler
        sampler_train = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        sampler_val = DistributedSampler(
            dataVal,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler_train,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        loaderVal = DataLoader(
            dataVal,
            batch_size=batch_size,
            sampler=sampler_val,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        loaderVal = DataLoader(
            dataVal,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

    if is_main_process:
        ema = EMA(model, beta=0.995, update_every=10)
        ema.to(device)
        ema.ema_model.eval()

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    load_weight = False  # True#
    if load_weight == True:
        model_string_name = 'deit'  # args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_dir + '/best.pt', map_location='cpu')
        # #
        # model.load_state_dict(state_dict["model"])

        # checkpoint = torch.load(weight, map_location='cpu')
        current_epoch = checkpoint["epoch"] + 1
        # model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        # epochs = 0
        if HAS_TPU:
            print('load weight')
        else:
            accelerator.print('load weight')

        model.load_state_dict(checkpoint['model'])
        if is_main_process:
            ema = EMA(model, beta=0.995, update_every=10)
            ema.to(device)
            ema.ema_model.load_state_dict(checkpoint['model'])
            ema.ema_model.eval()

        # 只在 GPU 模式下使用 accelerator.prepare
        if not HAS_TPU:
            model, opt, loader, loaderVal = accelerator.prepare(model, opt, loader, loaderVal)
    else:
        current_epoch = 0
        # 只在 GPU 模式下使用 accelerator.prepare
        if not HAS_TPU:
            model, opt, loader, loaderVal = accelerator.prepare(model, opt, loader, loaderVal)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    best_loss = 1000000
    Loss_tr = []
    Loss_val = []
    model.train()

    for epoch in range(current_epoch, current_epoch + args.epochs):

        # logger.info(f"Beginning epoch {epoch}...")
        for y, y_st, x in loader:

            x = x.to(device)  # image
            y = y.to(device)  # voltage
            y_st = y_st.to(device)

            # batch_mask = x != 0
            # x = torch.cat([x, batch_mask], dim=1)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, y_st=y_st)

            # 直接前向传播，不使用 autocast（避免 XLA 张量生命周期问题）
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"]

            opt.zero_grad()
            # Backward pass - 根据设备类型选择不同的方法
            if HAS_TPU:
                loss.backward()
            else:
                accelerator.backward(loss)
                accelerator.wait_for_everyone()

            # 梯度裁剪
            if HAS_TPU:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            else:
                accelerator.clip_grad_norm_(model.parameters(), 1)

            opt.step()

            if is_main_process:
                ema.update()
            # scheduler.step()
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            #if train_steps>11:break
            if train_steps % args.log_every == 0:
                logger.info('*' * 40)
                # Measure training speed:

                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)

                # avg_loss = avg_loss.item()
                # if is_main_process:
                if HAS_TPU:
                    # TPU 单节点，不需要 gather
                    avg_loss = avg_loss.item()
                else:
                    avg_loss = accelerator.gather(avg_loss)
                    avg_loss = avg_loss.mean().item()

                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:

                running_loss = 0
                log_steps = 0
                start_time = time()
                Loss_tr.append(avg_loss)

                # Save DiT checkpoint:
            # exit()
            if train_steps % args.ckpt_every == 0:

                model.eval()
                val_loss_v = 0
                log_steps_v = 0
                with torch.no_grad():
                    for y, y_st, x in loaderVal:
                        x = x.to(device)
                        y = y.to(device)
                        y_st = y_st.to(device)

                        # batch_mask = x != 0
                        # x = torch.cat([x, batch_mask], dim=1)
                        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                        model_kwargs = dict(y=y, y_st=y_st)

                        # 直接计算，不使用 autocast
                        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                        loss = loss_dict["loss"].mean()

                        val_loss_v += loss.item()
                        log_steps_v += 1

                    val_loss_v = torch.tensor(val_loss_v / log_steps_v, device=device)

                    # val_loss_v = val_loss_v.item()
                    if HAS_TPU:
                        # TPU 单节点，不需要 gather
                        val_loss_v = val_loss_v.item()
                    else:
                        val_loss_v = accelerator.gather(val_loss_v)
                        val_loss_v = val_loss_v.mean().item()

                    logger.info(
                        f"(step={train_steps:07d}) Valid Loss: {val_loss_v:.4f}")
                    Loss_val.append(val_loss_v)
                    if val_loss_v < best_loss:
                        best_loss = val_loss_v
                        if is_main_process:
                            checkpoint = {
                                "model": ema.ema_model.state_dict(),
                                # "model":  model.state_dict(),
                                # "optimizer": opt.state_dict(),
                                "epoch": epoch
                            }
                            checkpoint_path = f"{checkpoint_dir}/best.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                model.train()

    if is_main_process:
        sio.savemat(checkpoint_dir + '/loss1.mat',
                    {'loss_stage1Tr': np.stack(Loss_tr),
                     'loss_stage1Val': np.stack(Loss_val)})


def test(args):
    # 初始化设备和加速器
    if HAS_TPU:
        # TPU 模式 - 使用正确的 torch-xla API
        import torch_xla
        device = torch_xla.device()  # 替代已弃用的 xm.xla_device()
        accelerator = None
        mixed_precision_mode = 'bf16'
        print(f"✅ 使用 TPU 设备: {device}")
        gpus = 8  # Kaggle TPU v5e-8
    else:
        # GPU 模式，使用 Accelerator
        mixed_precision_mode = 'fp16'
        accelerator = Accelerator(mixed_precision=mixed_precision_mode)
        device = accelerator.device
        print(f"✅ 使用 GPU 设备: {device}")
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    seed = args.global_seed
    init_seed(seed)

    # 判断是否为主进程（TPU 和 GPU 都需要）
    if HAS_TPU:
        is_main_process = True
    else:
        is_main_process = accelerator.is_local_main_process

    model_string_name = 'deit'  # args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    # os.makedirs(checkpoint_dir, exist_ok=True)
    # logger = create_logger(experiment_dir)
    # logger.info(f"Experiment directory created at {experiment_dir}")

    model = DiT().to(device)

    # TPU：不直接转换模型，使用 autocast 代替
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    # model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")

    # model_name='DEIT'
    # state_dict = torch.load(checkpoint_dir + '/best.pt')

    state_dict = torch.load(checkpoint_dir + '/best.pt', map_location='cpu')
    # print(model.y_embedder)
    # model.load_state_dict(state_dict["model"])
    # print(state_dict["epoch"])
    model.load_state_dict(state_dict['model'])

    # 获取设备名称
    if HAS_TPU:
        gpuname = "TPU"
    elif torch.cuda.is_available():
        gpuname = torch.cuda.get_device_name(0)
    else:
        gpuname = "CPU"
    # print(gpuname)
    modelname = 'DEIT'

    # 使用命令行参数或默认数据路径，如果指定路径不存在则回退到'./data'
    datapath = get_data_path(args.data_path)
   
 
    path = datapath + '/test/'
    dataTe = EITdataset(path, modelname, dataset='data', backup_data_path='./data')
    if args.data == 'uef2017':
        path = datapath + '/data2017/'
        dataTe = EITdataset(path, modelname, dataset='data2017', backup_data_path='./data')
    elif args.data == 'ktc2023':
        path = datapath + '/data2023/'
        dataTe = EITdataset(path, modelname, dataset='data2023', backup_data_path='./data')
    loaderTe = DataLoader(
        dataTe,
        batch_size=args.global_batch_size * 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    # 只在 GPU 模式下使用 accelerator.prepare
    if not HAS_TPU:
        model, loaderTe = accelerator.prepare(model, loaderTe)

    if HAS_TPU:
        print('sampling steps', args.samplingsteps)
    else:
        accelerator.print('sampling steps', args.samplingsteps)
    model.eval()
    with torch.no_grad():
        pred = []
        gt1 = []
        RMSE = []
        for i, (y, y_st, x) in enumerate(loaderTe):
            x = x.to(device)
            y = y.to(device)
            y_st = y_st.to(device)

            # print(x.shape)
            # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, y_st=y_st)

            # batch_mask = x != 0
            # x = torch.cat([x, batch_mask], dim=1)
            shape = x.shape
            # shape = [x.shape[0], 4, 16, 16]

            # 直接推理，不使用 autocast
            if gpus == 1 or HAS_TPU:
                out = diffusion.ddim_sampleEIT(model, shape, args.samplingsteps, model_kwargs)
            else:
                out = diffusion.ddim_sampleEIT(model.module, shape, args.samplingsteps, model_kwargs)

            # print(out.shape)
            if not HAS_TPU:
                out = accelerator.gather(out)
            # out = out[:, 0:1, :, :]
            #

            # plt.subplot(121)
            # plt.imshow(x[0, 0].cpu())
            # plt.subplot(122)
            # plt.imshow(out[0, 0].cpu())
            # plt.show()
            if not HAS_TPU:
                x = accelerator.gather(x)

            rmse = (x - out).square().mean().sqrt()
            if HAS_TPU:
                print('out', i, out.shape, 'rmse: ', rmse)
            else:
                accelerator.print('out', i, out.shape, 'rmse: ', rmse)
            # while 1: pass
            RMSE.append(rmse)
            out = out.squeeze()
            x = x.squeeze()
            # accelerator.wait_for_everyone()
            # accelerator.print(type(gt1),gt1)
            # accelerator.wait_for_everyone()
            gt1.append(x)
            pred.append(out)

        pred = torch.cat(pred, dim=0)
        gt1 = torch.cat(gt1, dim=0)
        if HAS_TPU:
            print('out', pred.shape)
        else:
            accelerator.print('out', pred.shape)
        RMSE = torch.stack(RMSE, dim=0)
        # accelerator.print('average RMSE', RMSE.mean())
        rmse = (gt1 - pred).square().mean().sqrt()
        pred1=pred.clone()
        pred = pred / 2 + 0.5
        gt1 = gt1 / 2 + 0.5
        max1, _ = torch.max(gt1, 1)
        max1, _ = torch.max(max1, 1)
        psnr = 10 * torch.log10(max1.square() / ((gt1 - pred).square().mean([1, 2]) + 1e-12))
        if HAS_TPU:
            print('PSNR ', psnr.mean())
            print('RMSE whole ', rmse)
        else:
            accelerator.print('PSNR ', psnr.mean())
            accelerator.print('RMSE whole ', rmse)

        torch.save(psnr.mean(),checkpoint_dir + '/'+'PSNR.pt' )
        # pred1 = pred.clone()
        # pred1[ind] = pred
        # pred = pred1
        if is_main_process or HAS_TPU:
            sio.savemat(checkpoint_dir + '/' + modelname + '.mat',
                        {'pred': pred1.cpu() * dataTe.current / dataTe.voltage})
            # sio.savemat(checkpoint_dir + '/' +     'GT2023.mat',
            # {'GT': gt1.cpu()})


if __name__ == "__main__":

    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--data", type=str, choices=["simulated", "uef2017", "ktc2023"], default="simulated")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--samplingsteps", type=int, default=5)
    args = parser.parse_args()

    if args.mode == 'train':
        if HAS_TPU:
            # TPU 训练：torch_xla 会自动使用所有可用的 TPU 核心
            # Kaggle 环境不支持 xmp.spawn()，但单进程会自动并行使用 8 个核心
            print(f"🚀 启动 TPU 训练（自动使用所有可用核心）")
            print(f"   全局批大小: {args.global_batch_size}")
            print(f"   TPU 会自动并行使用 8 个核心")
            main(args)
        else:
            # GPU 单进程训练
            main(args)

    elif args.mode == 'test':
        test(args)
   

