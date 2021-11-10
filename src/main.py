import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models
import torch.nn.init as init

import csv


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.hardnext import HarDNeXt



model_names = ['hardnext']

arch_list = ['28', '32', '39', '50', '56']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='/dataset/imagenet/raw_data/',
                    help='path to dataset')
parser.add_argument('--model_name', default='HardneXt', choices=model_names)
parser.add_argument('-a', '--arch', metavar='ARCH', default='50', choices=arch_list)
parser.add_argument('-dw', '--depthwise', action='store_true')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--warmup', default=0, type=int, help='number of warmup epochs to run')

parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

train_epoch = 0
train_acc1 = 0
train_acc5 = 0
train_loss = 0

val_acc1 = 0
val_acc5 = 0
val_loss = 0

learning_rate = 0
full_model_name = ""

writer = SummaryWriter('tb-logs')

def main():

    global full_model_name

    #torch.backends.cudnn.enabled=False

    args = parser.parse_args()

    # tensorboard summarywriter
    

    #depth_wise = args.depthwise #'ds' in args.arch
    arch = int(args.arch) #int(args.arch[7:9])


    if args.depthwise:
        full_model_name = args.model_name+"("+args.arch+"DS)"
    else:
        full_model_name = args.model_name+"("+args.arch+")"
    
    print(full_model_name)

    # 開啟輸出的 CSV 檔案
    with open(full_model_name + '_'+str(args.epochs)+'_'+str(args.batch_size)+'_'+str(args.lr)+'_training_log.csv', 'w', newline='') as csvfile:
        # 建立CSV檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow(['Epoch', 'training_acc1','training_acc5', 'training_loss', 'val_acc1', 'val_acc5', 'val_loss'])


    

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

    

def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key and 'norm' not in key:
                    init.xavier_normal_(m.state_dict()[key])
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    #global full_model_name

    if args.pretrained:
        print("Not implemented !!! QQ")
        
    else:
        model = HarDNeXt(arch=int(args.arch), depth_wise=args.depthwise)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif not args.pretrained:
        model.apply(weights_init)
        
    total_params = sum(p.numel() for p in model.parameters())

    print( "Parameters=", total_params )
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        ''' for per epoch warmup'''
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        

        

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # tensorboard summary write
        writer.add_scalars('Loss/',
                            {'train_loss': train_loss,
                            'val___loss': val_loss}, epoch)
        
        writer.add_scalars('Top-1 Acc/',
                            {'train_Acc': train_acc1,
                            'val___Acc': val_acc1}, epoch)
        
        writer.add_scalars('Top-5 Acc/',
                            {'train_loss': train_acc5,
                            'val___loss': val_acc5}, epoch)
        
        writer.add_scalars('Learning Rate',
                            {'lr': learning_rate}, epoch)

        

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    #data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    lrm = ConstantMeter('lr')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1, top5, lrm,
                              prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        #data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        ''' for per batch warmup'''
        # # adjust learning rate
        # adjust_learning_rate(optimizer, epoch, args)


        lrm.update(optimizer.param_groups[0]['lr'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


        global train_epoch 
        train_epoch = epoch
        global train_acc1 
        train_acc1 = top1.avg.item()
        global train_acc5 
        train_acc5 = top5.avg.item()
        global train_loss 
        train_loss = losses.avg

    

def validate(val_loader, model, criterion, args):

    global full_model_name
    
    print(train_epoch, train_acc1, train_acc5, train_loss)
    
    batch_time = AverageMeter('Time', ':6.3f', avg=False)
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
    
    global val_acc1, val_acc5, val_loss

    val_acc1 = top1.avg.item()
    val_acc5 = top5.avg.item()
    val_loss = losses.avg
    
    # 開啟輸出的 CSV 檔案
    with open(full_model_name + '_'+str(args.epochs)+'_'+str(args.batch_size)+'_'+str(args.lr)+'_training_log.csv', 'a', newline='') as csvfile:
        # 建立CSV檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow([train_epoch, train_acc1, train_acc5, train_loss, top1.avg.item(), top5.avg.item(), losses.avg])

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.display_avg = avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        if self.display_avg:
          fmtstr = '{name}: {avg' + self.fmt + '}'
        else:
          fmtstr = '{name}: {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ConstantMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
    
    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = '{name}: {val:f})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

init_step = 0  # for warmup learning rate
decay_init_step = 0 # for the rest decay part

def adjust_learning_rate(optimizer, epoch, args):
    
    #Cosine learning rate decay
    ''' Per Batch warmup '''
    total_data = 1281167 # imageNet training data
    step = total_data // args.batch_size + 1 # drop_liat == False
    global init_step, decay_init_step, learning_rate

    total_warmup_step = args.warmup*step

    if epoch < args.warmup and args.warmup != 0:
        init_step += 1
        lr = (args.lr * init_step) /args.warmup

    else:
        
        

        # per epoch cosine decay (HarDNet original)
        lr = 0.5 * args.lr  * (1 + np.cos(np.pi * (epoch-args.warmup)/ (args.epochs-args.warmup)))

        ## per batch cosine decay (me)
        # total_decay_step = (args.epochs-args.warmup)*step

        # lr = 0.5 * args.lr  * (1 + np.cos(np.pi * (decay_init_step)/ (total_decay_step)))
        # decay_init_step += 1
    
    
    # lr = 0.5 * args.lr  * (1 + np.cos(np.pi * (epoch)/ args.epochs ))

    learning_rate = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
