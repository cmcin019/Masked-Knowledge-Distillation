# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from os import system
import sys
import datetime
import time
import math
import matplotlib.pyplot as plt
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
# from tensorboardX import SummaryWriter
from models.head import iBOTHead
from loader import ImageFolderMask
from evaluation.unsupervised.unsup_cls import eval_pred
from evaluation.eval_linear import eval_linear

def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                 'swin_tiny','swin_small', 'swin_base', 'swin_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use activation in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
        
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs of training.') # epochs
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=0, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='./data/CIFAR10/imagenet32/res', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--loss_functions", default='van', type=str, nargs='*', help="van ,dis, ang, sim")
    return parser

def train_ibot(args):
    # utils.init_distributed_mode(args)
    # utils.fix_random_seeds(args.seed)
    # print("git:\n  {}\n".format(utils.get_sha()))
    # print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    # cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    dataset = ImageFolderMask(
        args.data_path, train=True, download=True, 
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch) if args.data_path == './data/CIFAR10' else ImageFolderMask(
            args.data_path,
            transform=transform,
            patch_size=pred_size,
            pred_ratio=args.pred_ratio,
            pred_ratio_var=args.pred_ratio_var,
            pred_aspect_ratio=(0.3, 1/0.3),
            pred_shape=args.pred_shape,
            pred_start_epoch=args.pred_start_epoch)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    ) 
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True, 
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # print(student)
    # print(teacher)
    # return 0

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student, 
        iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False) if \
            'swin' in args.arch else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False) if \
        'swin' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # print(student)
    # print(teacher)
    # return 0
    
    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        # writer = SummaryWriter(logdir=local_runs)
        
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting iBOT training!")
    plot_info = {'loss': [], 'acc': [], 'title':[*args.loss_functions]}
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, plot_info, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint_{args.loss_functions}.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / f"logs/{args.loss_functions}/log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                # for k, v in train_stats.items():
                #     writer.add_scalar(k, v, epoch)


    utils.save_plot(plot_info['loss'], plot_path, 'Loss', plot_info['title'][0]+'_Loss')
    utils.save_plot(plot_info['acc'], plot_path, 'Accuracy', plot_info['title'][0]+'_Acc')
   
   
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    return plot_info

    # eval_linear()


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, plot_info, args):
    system('cls' if os.name == 'nt' else 'clear')
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    pred_labels, real_labels = [], []
    for it, (images, labels, masks) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]   
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output = teacher(images[:args.global_crops_number])
            student_output = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number])
            
            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(args.loss_functions, student_output, teacher_output, student_local_cls, masks, epoch)
            loss = all_loss.pop('loss')
            if it % 200 == 0:
                plot_info['loss'].append(loss.item())
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        if it % 200 == 0:
            plot_info['acc'].append(acc.item())
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.to(pred1.device)))

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc) 

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    return return_dict


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def _vanilla_loss(self, epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask):
        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
       
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2

        return total_loss1, total_loss2

    # From https://github.com/lenscloth/RKD/blob/0a6c3c0c190722d428322bf71703c0ae86c25242/metric/utils.py#L6
    def pdist(self, e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        e = F.normalize(e, p=2, dim=-1)
        prod = e @ e.t()
        dist = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            dist = dist.sqrt()

        dist = dist.clone()
        dist[range(len(e)), range(len(e))] = 0

        mean_td = dist[dist>0].mean()
        dist= dist / mean_td

        return dist


    def _rk_dist_loss(self, epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask):
        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)

        # torch.Size([64, 8192])
        # print(teacher_cls_c.shape)

        # rk_t1_d = self.pdist(teacher_cls_c, squared=False)
        # rk_t2_d = self.pdist(teacher_cls_c[2:], squared=False)
        # print(rk_t1_d.shape)

        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        # print(teacher_cls_c[0] == teacher_cls_c[1])
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0

        # half = len(teacher_cls_c) // 2

        # print(student_mask[0].flatten(-2, -1).float())
        # print(student_mask[0].flatten(-2, -1).sum(dim=-1).clamp(min=1.0))


        # print(student_cls.shape)
        # print(student_patch.view(student_patch.size(0), -1).shape)
        # print(student_mask[0].shape)
        # print(teacher_cls.shape)
        # print(teacher_patch.view(teacher_patch.size(0), -1).shape)


        # TODO: Method combining all
        s_cls_d= self.pdist(student_cls, squared=False)
        s_patch_d = self.pdist(student_patch.view(student_patch.size(0), -1), squared=False)
        s_mask = torch.stack(student_mask, dim=0).view(len(student_mask) * student_mask[0].shape[0], -1)
        s_mask_d = self.pdist(s_mask.float(), squared=False) # This does not make sense 

        # print(student_patch.shape)
        # print(student_mask[0].shape)
        # print(s_mask_d.shape)


        t_cls_d= self.pdist(teacher_cls, squared=False)
        t_patch_d = self.pdist(teacher_patch.view(teacher_patch.size(0), -1), squared=False)

        # total_loss1 = F.smooth_l1_loss(t_cls_d, s_cls_d, reduction='mean')
        # total_loss2 = F.smooth_l1_loss(t_patch_d, s_patch_d, reduction='mean')

        total_loss1 = torch.sum(-t_cls_d * F.log_softmax(s_cls_d, dim=-1), dim=-1).mean() / (len(student_mask) * len(student_mask) )
        loss2 = torch.sum(-t_patch_d * F.log_softmax(s_patch_d, dim=-1), dim=-1)

        total_loss2 = torch.sum(loss2 * s_mask_d.float(), dim=-1) / s_mask_d.sum(dim=-1).clamp(min=1.0)
        total_loss2 = total_loss2.mean() / (len(student_mask) * len(student_mask) )

        # print(total_loss1)
        # print(total_loss2)

        # print(teacher_cls)
        # print(student_cls)
        # print(t_cls_d)
        # print(s_cls_d)

        # print(total_loss1) 
        # print(total_loss2)

        return total_loss1, total_loss2


        # # TODO: Method combining pairs
        # #         This way will allow for the criss cross loss calculation
        # #         Seen in the original iBOT method
        # # torch.Size([16, 8192])  # when crop num = 4
        # # print(teacher_cls_c[0].shape)

        # td_cls_1 = torch.sub(teacher_cls_c[0], teacher_cls_c[1]).pow(2)
        # td_cls_2 = torch.sub(teacher_cls_c[2], teacher_cls_c[3]).pow(2)

        # norm_td_cls_1 = F.normalize(td_cls_1, p=2, dim=-1)
        # norm_td_cls_2 = F.normalize(td_cls_2, p=2, dim=-1)

        # norms_td_cls = (norm_td_cls_1, norm_td_cls_2)

        # td_patch_1 = torch.sub(teacher_patch_c[0], teacher_patch_c[1]).pow(2)
        # td_patch_2 = torch.sub(teacher_patch_c[2], teacher_patch_c[3]).pow(2)

        # norm_td_patch_1 = F.normalize(td_patch_1, p=2, dim=-1)
        # norm_td_patch_2 = F.normalize(td_patch_2, p=2, dim=-1)

        # norms_td_patch = (norm_td_patch_1, norm_td_patch_2)

        # # print(norm_td_cls_1.shape)


        # # rk_t1_d = self.pdist(teacher_cls_c[:half], squared=False)
        # # rk_t2_d = self.pdist(teacher_cls_c[half:], squared=False)

        # # print(rk_t1_d.shape)
        # # print(teacher_cls_c[0].shape)


        # sd_cls_1 = torch.sub(student_cls_c[0], student_cls_c[1]).pow(2)
        # sd_cls_2 = torch.sub(student_cls_c[2], student_cls_c[3]).pow(2)

        # norm_sd_cls_1 = F.normalize(sd_cls_1, p=2, dim=-1)
        # norm_sd_cls_2 = F.normalize(sd_cls_2, p=2, dim=-1)

        # norms_sd_cls = (norm_sd_cls_1, norm_sd_cls_2)

        # sd_patch_1 = torch.sub(student_patch_c[0], student_patch_c[1]).pow(2)
        # sd_patch_2 = torch.sub(student_patch_c[2], student_patch_c[3]).pow(2)

        # norm_sd_patch_1 = F.normalize(sd_patch_1, p=2, dim=-1)
        # norm_sd_patch_2 = F.normalize(sd_patch_2, p=2, dim=-1)

        # norms_sd_patch = (norm_sd_patch_1, norm_sd_patch_2)


        for q in range(len(norms_td_cls)):
            for v in range(len(norms_sd_cls)):
                if v == q: # Patch tokens
                    loss2 = torch.sum(-norms_td_patch[q] * F.log_softmax(norms_sd_patch[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)

                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()

                    n_loss_terms2 += 1
                else: # Class token
                    loss1 = torch.sum(-norms_td_cls[q] * F.log_softmax(norms_sd_cls[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()

                    n_loss_terms1 += 1
            
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2



        return total_loss1, total_loss2
    
    def angle(self, e):
        res = (e.unsqueeze(0) - e.unsqueeze(1))
        norm_res = F.normalize(res, p=2, dim=2)
        res_angle = torch.bmm(norm_res, norm_res.transpose(1, 2))
        return res_angle

    def _rk_angle_loss(self, epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask):
        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)

        # torch.Size([64, 8192])
        # print(teacher_cls_c.shape)

        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        # print(teacher_cls_c[0] == teacher_cls_c[1])
        teacher_patch_c = F.softmax(
            (teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0

        # TODO: Method combining all
        s_cls_d = self.angle(student_cls)
        s_patch_d = self.angle(student_patch.view(student_patch.size(0), -1))
        s_mask = torch.stack(student_mask, dim=0).view(
            len(student_mask) * student_mask[0].shape[0], -1)
        # This does not make sense
        s_mask_d = self.angle(s_mask.float())

        t_cls_d = self.angle(teacher_cls)
        t_patch_d = self.angle(teacher_patch.view(teacher_patch.size(0), -1))

        total_loss1 = torch.sum(-t_cls_d * F.log_softmax(s_cls_d, dim=-1),
                                dim=-1).mean() / (len(student_mask) * len(student_mask))
        loss2 = torch.sum(-t_patch_d *
                          F.log_softmax(s_patch_d, dim=-1), dim=-1)

        total_loss2 = torch.sum(loss2 * s_mask_d.float(),
                                dim=-1) / s_mask_d.sum(dim=-1).clamp(min=1.0)
        total_loss2 = total_loss2.mean() / (len(student_mask) * len(student_mask))

        # print(total_loss1)
        # print(total_loss2)

        # print(teacher_cls)
        # print(student_cls)
        # print(t_cls_d)
        # print(s_cls_d)

        # print(total_loss1)
        # print(total_loss2)

        return total_loss1, total_loss2
        

    # From https://github.com/lenscloth/RKD/blob/0a6c3c0c190722d428322bf71703c0ae86c25242/metric/utils.py#L6
    def psim(self, e, eps=1e-12):
        e_sum = e.sum(dim=1)
        e = F.normalize(e, p=2, dim=-1)
        prod = e @ e.t()
        # print(e_sum.shape, prod.shape)
        dot_prod = F.cosine_similarity(e_sum, 2 * prod)
        # print(dot_prod.shape)
        dist = dot_prod.clamp(min=eps)
        # print(dist.shape)
        dist = dist.clone()
        dist[range(len(e)), range(len(e))] = 0

        mean_td = dist[dist>0].mean()
        dist= dist / mean_td

        return dist


    def _rk_sim_loss(self, epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask):
        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)

        # torch.Size([64, 8192])
        # print(teacher_cls_c.shape)

        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        # print(teacher_cls_c[0] == teacher_cls_c[1])
        teacher_patch_c = F.softmax(
            (teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0

        # TODO: Method combining all
        s_cls_d = self.psim(student_cls)
        s_patch_d = self.psim(student_patch.view(student_patch.size(0), -1))
        s_mask = torch.stack(student_mask, dim=0).view(
            len(student_mask) * student_mask[0].shape[0], -1)
        # This does not make sense
        s_mask_d = self.psim(s_mask.float())

        t_cls_d = self.psim(teacher_cls)
        t_patch_d = self.psim(teacher_patch.view(teacher_patch.size(0), -1))

        total_loss1 = torch.sum(-t_cls_d * F.log_softmax(s_cls_d, dim=-1),
                                dim=-1).mean() / (len(student_mask) * len(student_mask))
        loss2 = torch.sum(-t_patch_d *
                          F.log_softmax(s_patch_d, dim=-1), dim=-1)

        total_loss2 = torch.sum(loss2 * s_mask_d.float(),
                                dim=-1) / s_mask_d.sum(dim=-1).clamp(min=1.0)
        total_loss2 = total_loss2.mean() / (len(student_mask) * len(student_mask))

        # print(total_loss1)
        # print(total_loss2)

        # print(teacher_cls)
        # print(student_cls)
        # print(t_cls_d)
        # print(s_cls_d)

        # print(total_loss1)
        # print(total_loss2)

        return total_loss1, total_loss2


    def forward(self, loss_functions, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        # # torch.Size([32, 8192]) torch.Size([32, 16, 8192])
        # print(student_cls.shape, student_patch.shape)
        # print(teacher_cls.shape, teacher_patch.shape)


        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_patch = student_patch / self.student_temp

        # total_loss1, total_loss2 = self._vanilla_loss(epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask)
        # # print(total_loss1.detach(), total_loss2.detach())
        # # total_loss1, total_loss2 = self._rk_angle_loss(epoch, teacher_cls, teacher_patch, student_cls_c, student_patch_c, student_mask)
        # # total_loss1, total_loss2 = self._rk_dist_loss(epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask)
        
        total_loss1, total_loss2 = 0, 0
        van_loss, dist_loss, ang_loss, sim_loss = {}, {}, {}, {}
        if 'van' in loss_functions:
            van_loss1, van_loss2 = self._vanilla_loss(
                epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask)
            total_loss1 += van_loss1
            total_loss2 += van_loss2
            van_loss = dict(van_cls=van_loss1, van_patch=van_loss2, van_loss=van_loss1 + van_loss2)

        if 'dis' in loss_functions:
            dist_loss1, dist_loss2 = self._rk_dist_loss(
                epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask)
            total_loss1 += dist_loss1
            total_loss2 += dist_loss2
            dist_loss = dict(dist_cls=dist_loss1, dist_patch=dist_loss2, dist_loss=dist_loss1 + dist_loss2)

        if 'ang' in loss_functions:
            ang_loss1, ang_loss2 = self._rk_angle_loss(
                epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask)
            total_loss1 += ang_loss1
            total_loss2 += ang_loss2
            ang_loss = dict(ang_cls=ang_loss1, ang_patch=ang_loss2, ang_loss=ang_loss1 + ang_loss2)
        
        if 'sim' in loss_functions:
            sim_loss1, sim_loss2 = self._rk_sim_loss(
                epoch, teacher_cls, teacher_patch, student_cls, student_patch, student_mask)
            total_loss1 += sim_loss1
            total_loss2 += sim_loss2
            sim_loss = dict(sim_cls=sim_loss1, sim_patch=sim_loss2, sim_loss=sim_loss1 + sim_loss2)

        # total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2, **van_loss, **dist_loss, **ang_loss, ** sim_loss)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])


    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        return crops # List of augmentations

if __name__ == '__main__':
    torch.cuda.empty_cache()
    plot_path = "out/plots/"
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    if args.loss_functions == ['test']:
        fig_loss, ax_loss = plt.subplots()
        fig_acc, ax_acc = plt.subplots()
        for x in [(['van'],128),(['dis'],128),(['ang'],16)]:
            torch.cuda.empty_cache()
            args.loss_functions = x[0]
            args.batch_size_per_gpu = x[1]
            plot_info_ = train_ibot(args)
            ax_loss.plot([n*200 for n in range(len(plot_info_['loss']))], plot_info_['loss'], label=x[0][0])
            ax_acc.plot([n*200 for n in range(len(plot_info_['acc']))], plot_info_['acc'], label=x[0][0])
            # dist.destroy_process_group()
        ax_loss.legend()
        ax_acc.legend()
        ax_loss.set_title('All_Loss')
        ax_acc.set_title('All_Accuracy')
        ax_loss.set(xlabel='Iterations', ylabel='Loss')
        ax_acc.set(xlabel='Iterations', ylabel='Loss')
        fig_loss.savefig( f"{plot_path}{'all_loss'}.png", bbox_inches='tight', dpi=150)
        fig_acc.savefig( f"{plot_path}{'all_accuracy'}.png", bbox_inches='tight', dpi=150)
        plt.close()
    else:
        _ = train_ibot(args)
