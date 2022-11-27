import utils
import datetime
import time
from tqdm import tqdm 
from os import system
import os
import matplotlib.pyplot as plt
import numpy as np

import datasets as dsets
import models.vision_transformer as vits
from models.head import iBOTHead
import parser 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

args = parser.get_args_parser()

def train_MRKD(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    # print("git:\n  {}\n".format(utils.get_sha()))
    # print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    train_loader, val_loader, _ = dsets.get_cifar10_dataloaders(args)

    student = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,  # stochastic depth
    )
    teacher = vits.__dict__[args.arch](
        patch_size=args.patch_size
    )
    embed_dim = student.embed_dim

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
        )
    )

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
    # else:
    #     # teacher_without_ddp and teacher are the same thing
    #     teacher_without_ddp = teacher
    # student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False) if \
    #     'swin' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # # teacher and student start with the same weights
    # teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    params_groups = utils.get_params_groups(student)
    optimizer = optim.SGD(params_groups, lr=0, momentum=0.9)

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(train_loader))

    print(f"Loss, optimizer and schedulers ready.")

    start_time = time.time()
    accuracy = 0
    for epoch in range(2):
        system('cls' if os.name == 'nt' else 'clear')
        print(f'Epoch {epoch}: {accuracy}')
        # ============ training one epoch... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp,  nn.CrossEntropyLoss(),
            train_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, None, args)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print(model_accuracy(student, val_loader))

def train_one_epoch(student, teacher, teacher_without_ddp, MRKD_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    for _, ((data, masks), targets) in enumerate(tqdm(data_loader)):
        data = [image.cuda(non_blocking=True) for image in data]
        masks = [msk.cuda(non_blocking=True) for msk in masks]  

        teacher_output = teacher(data[:args.global_crops_number])
        student_output = student(data[:args.global_crops_number], mask=masks[:args.global_crops_number])

        # for image, mask in zip(data, masks):
        #     targets = targets.cuda()
        #     scores_s = student(image)
        #     # scores_t = teacher_without_ddp(d)
        #     loss = MRKD_loss(scores_s, targets)
            
        #     optimizer.zero_grad()
        #     loss.backward()
            
        #     optimizer.step()

        #     print(masks)
        #     # print(image)

        #     np_images = torchvision.utils.make_grid(image.cpu().data, normalize=True).numpy()
        #     fig, ax = plt.subplots()
        #     ax.imshow(np.transpose(np_images,(1,2,0)))
        #     fig.savefig(f'images/images.jpg', bbox_inches='tight', dpi=150)

        #     np_masks = torchvision.utils.make_grid(mask.cpu().data, normalize=True).numpy()
        #     fig, ax = plt.subplots()
        #     ax.imshow(np.transpose(np_masks,(1,2,0)))
        #     fig.savefig(f'images/masks.jpg', bbox_inches='tight', dpi=150)
        #     break
        # break

    return 0







def model_accuracy(model, loader):
	correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for image, labels in loader:
			image = image.cuda()
			labels = labels.cuda()

			outputs = model(image)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted.to(device='cpu')==labels.to(device='cpu')).sum().item()
		TestAccuracy = 100 * correct / total

	model.train()
	return(TestAccuracy)






train_MRKD(args)