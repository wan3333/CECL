import os
import argparse
import builtins
import random
import shutil
import time
import warnings
from sqlalchemy import true
import torch
import torch.nn.functional as F
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data.distributed
import tensorboard_logger as tb_logger

from torch.optim.lr_scheduler import CosineAnnealingLR


# from tensorboardX import SummaryWriter as Writer
from utils.model import CECL
from tqdm import tqdm
from backbone.net import *
from utils.builder import build_cifarn_dataset_loader
from utils.util import Config, load_from_cfg, linear_rampup2
from utils.utils_algo import *
from utils.utils_loss import SupConLoss, EntropyLoss, CE_Soft_Label
# from utils.cifar100_OOD import load_cifar100_ood

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation')

parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='architecture')
parser.add_argument('--num-class', default=80, type=int,
                    help='number of class')
parser.add_argument('--moco_queue', default=8192, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--sigma', default=0.2, type=float,
                    help='SGD noise')


flag = 0

def main():
    args = parser.parse_args()
    # cfg_file = 'config/cifar100.cfg'
    cfg = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        cfg.set_item(item, args.__dict__[item])


    iterations = cfg.lr_decay_epochs.split(',')
    cfg.lr_decay_epochs = list([])
    for it in iterations:
        cfg.lr_decay_epochs.append(int(it))
    cfg.num_class = int(cfg.n_classes * (1-cfg.openset_ratio))

    if cfg.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if cfg.gpu != -1:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    model_path = 'ds_{ds}_nr_{nr}_or_{osr}_lr_{lr}_ep_{ep}_lw_{lw}_pm_{pm}_arch_{arch}_opt_{opt}_sd_{seed}'.format(
        ds=cfg.dataset,
        nr=cfg.closeset_ratio,
        osr=cfg.openset_ratio,
        lr=cfg.lr,
        ep=cfg.epochs,
        lw=cfg.loss_weight,
        pm=cfg.proto_m,
        arch=cfg.arch,
        opt=cfg.opt,
        seed=cfg.seed
    )
    cfg.exp_dir = os.path.join(cfg.exp_dir, model_path)
    if not os.path.exists(cfg.exp_dir):
        os.makedirs(cfg.exp_dir)

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node   ', ngpus_per_node)
    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(cfg.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    cudnn.benchmark = True
    cfg.gpu = gpu
    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
        cudnn.deterministic = True
    if cfg.gpu != -1:
        print("Use GPU: {} for training".format(cfg.gpu))
    # suppress printing if not master
    if cfg.multiprocessing_distributed and cfg.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass
    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)

    # create model
    model = CECL(cfg, SupConNet)

    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu != 'None':
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int((cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif cfg.gpu != 'None':
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # set optimizer
    if cfg.opt == 'adam':
        optimizer_model = torch.optim.Adam(model.parameters(), cfg.lr, betas=(0.9, 0.999))
        # optimizer_model = torch.optim.Adam(model.parameters(), cfg.lr)
    else:
        optimizer_model = torch.optim.SGD(model.parameters(), cfg.lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)

    # scheduler = CosineAnnealingLR(optimizer_model, cfg.epochs, cfg.lr/100)


    # optionally resume from a checkpoint
    if cfg.resume != -1:
        print(cfg.resume)
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu == -1:
                checkpoint = torch.load(cfg.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_model.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))


    dataset1, train_loader1, train_sampler1, dataset2, train_loader2, train_sampler2, test_loader  = build_cifarn_dataset_loader(cfg)


    loss_fn = nn.CrossEntropyLoss()
    loss_fn_soft = CE_Soft_Label()
    loss_cont_fn = SupConLoss()


    if cfg.gpu == 0:
        logger = tb_logger.Logger(logdir=os.path.join(cfg.exp_dir, 'tensorboard'), flush_secs=2)
    else:
        logger = None

    print('\nStart Training\n')

    best_acc = 0

    for epoch in range(1, cfg.epochs+1):
        if epoch <= cfg.warm:
            train_loader = train_loader1
            train_sampler = train_sampler1
        else:
            train_loader = train_loader2
            train_sampler = train_sampler2

        is_best = False
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(cfg, optimizer_model, epoch-1)

        if epoch <= cfg.warm:
            # warm-up
            warm(train_loader, model, loss_fn_soft, loss_cont_fn, optimizer_model, epoch, logger, cfg)
        else:
            # training
            train(train_loader, model, loss_fn_soft, loss_cont_fn, optimizer_model, epoch, logger, cfg)
        # test
        acc_test = test(test_loader, model, epoch, logger, cfg)

        # scheduler.step()

        with open(os.path.join(cfg.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Acc {}, Best Acc {}. (lr {})\n'.format(epoch
                                                                    , acc_test, best_acc,
                                                                    optimizer_model.param_groups[0]['lr']))
        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        if not cfg.multiprocessing_distributed or (cfg.multiprocessing_distributed
                                                   and cfg.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'arch': cfg.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer_model.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(cfg.exp_dir),
                best_file_name='{}/checkpoint_best.pth.tar'.format(cfg.exp_dir))


def step_flagging(content):
    print('=================================================')
    print(content, flush=True)
    print('=================================================')



def warm(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, tb_logger, cfg=None):
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [acc_cls, loss_cls_log, loss_cont_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # --------------- start ---------------
    for i, (index, D, train_Y, true_Y) in enumerate(train_loader):
        X, X_w, X_s = D
        train_Y = train_Y.long().detach().cuda()

        # for showing training accuracy and will not be used when training
        true_Y = true_Y.long().detach().cuda()
        batch_size = train_Y.size(0)


        global flag
        if flag == 0:
            step_flagging(f'start the warm-up step for {cfg.warm} epochs.')
            flag += 1

        output_x, moco_queue = model(X, X_w, train_Y, None, None, cfg, 's0')
        queue_features = moco_queue['feature']
        queue_targets = moco_queue['target']

        
        mask = torch.eq(queue_targets[:batch_size].unsqueeze(dim=1), queue_targets.unsqueeze(dim=1).T).float().cuda()

        y = torch.zeros(batch_size, cfg.num_class).cuda().scatter_(1, train_Y.view(-1, 1), 1)
        loss_cls = loss_fn(output_x, y).mean()
        loss_cont = loss_cont_fn(features=queue_features, index=torch.ones(batch_size).bool(), mask=mask, batch_size=batch_size)
        loss = loss_cls + cfg.weight_cl * loss_cont

        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.item())

        # log accuracy
        acc = accuracy(output_x, true_Y)[0]
        acc_cls.update(acc[0])


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % cfg.print_freq == 0:
            progress.display(i)

    if cfg.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)

def train(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, tb_logger, cfg=None):
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [acc_cls, loss_cls_log, loss_cont_log],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    # --------------- start ---------------
    for i, (index, D, train_Y, true_Y, nc, c) in enumerate(train_loader):
        X, X_w, X_s = D
        train_Y = train_Y.long().detach().cuda()
        nc, c = nc.cuda(), c.cuda()
        batch_size = train_Y.size(0)

        
        # for showing training accuracy and will not be used when training
        true_Y = true_Y.long().detach().cuda()

        global flag
        if flag == 1:
            step_flagging('start the robust learning step.')
            flag += 1
        output_x, selected, moco_queue = model(X, X_w, train_Y, nc, c, cfg, 's1')


        queue_features = moco_queue['feature']
        queue_targets = moco_queue['target']
        queue_isid = moco_queue['IDindex']


        mask = torch.eq(queue_targets[:batch_size].unsqueeze(dim=1), 
        queue_targets[batch_size:].unsqueeze(dim=1).T).float().cuda().mul(queue_isid[batch_size:])

        y = torch.zeros(batch_size, cfg.num_class).cuda().scatter_(1, train_Y.view(-1, 1), 1)
        loss_cls = loss_fn(output_x[selected], y[selected]).mean()

        # loss_cls = loss_fn(output_x[selected], train_Y[selected])
        # loss_cont = loss_cont_fn(features=queue_features, index=selected, mask=mask, batch_size=batch_size)
        loss_cont = loss_cont_fn(features=queue_features, index=selected, mask=mask, batch_size=batch_size)

        loss = loss_cls + cfg.weight_cl * loss_cont
        # loss = loss_cls


        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.item())

        # log accuracy
        with torch.no_grad():
            ood = true_Y == -1
            s = selected==True * (~ood)

        acc = accuracy(output_x[s], true_Y[s])[0]
        acc_cls.update(acc[0])


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % cfg.print_freq == 0:
            progress.display(i)
    



    if cfg.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)


def test(test_loader, model, epoch, tb_logger, cfg):
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        for i, (index, X, true_Y) in enumerate(test_loader):

            X, true_Y = X.cuda(), true_Y.cuda()
            outputs = model(X, mode='test')
            acc1, acc5 = accuracy(outputs, true_Y, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

        # average across all processes
        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(cfg.gpu)
        dist.all_reduce(acc_tensors)
        acc_tensors /= cfg.world_size

        print('Accuracy is %.2f%% (%.2f%%) \n' % (acc_tensors[0], acc_tensors[1]))
        if cfg.gpu == 0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('Top5 Acc', acc_tensors[1], epoch)
    return acc_tensors[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


if __name__ == '__main__':
    main()
