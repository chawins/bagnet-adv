'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from bagnet import BottleneckNoBn, SmallBagNet, TinyBagNet
from resnet import Bottleneck
from utils import *

# def avg_patches(net, criterion, inputs, targets, patch_size, stride):
#
#     num_patches = inputs.size()[2] // stride
#     pad = torch.nn.ZeroPad2d(patch_size // 2)
#     padded = pad(inputs)
#
#     outputs = torch.empty([128, num_patches**2, 10],
#                           dtype=torch.float32, device='cuda')
#     for y in range(num_patches):
#         for x in range(num_patches):
#             out = net(padded[:, :, y * stride:y * stride + patch_size,
#                              x * stride:x * stride + patch_size])
#             outputs[:, y * num_patches + x] = out
#
#     outputs = torch.stack(outputs_list, 1)
#     mean_outputs = outputs.mean(1)
#     loss = criterion(mean_outputs, targets)
#     return mean_outputs, outputs, loss


def evaluate(net, dataloader, criterion, device):

    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # mean_outputs, _, loss = avg_patches(
            #     net, criterion, inputs, targets, 8, 4)
            # outputs = net(inputs).mean(1)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    return val_loss / val_total, val_correct / val_total


def train(net, trainloader, validloader, criterion, optimizer, epoch, device,
          log, save_best_only=True, best_acc=0, model_path='./model.pth'):

    net.train()
    # net.eval()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # mean_outputs, _, loss = avg_patches(
        #     net, criterion, inputs, targets, 8, 4)
        # outputs = net(inputs).mean(1)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    # val_loss, val_acc = evaluate(net, trainloader, criterion, device)
    # print(val_loss, val_acc)
    val_loss, val_acc = evaluate(net, validloader, criterion, device)

    log.info(' %5d | %.4f, %.4f | %8.4f, %7.4f', epoch,
             train_loss / train_total, train_correct / train_total,
             val_loss, val_acc)

    # Save model weights
    if not save_best_only or (save_best_only and val_acc > best_acc):
        log.info('Saving model...')
        torch.save(net.state_dict(), model_path)
        best_acc = val_acc
    return best_acc


def main():

    # Set experiment id
    exp_id = 15
    filename = 'train_bagnet'

    # Training parameters
    batch_size = 256
    epochs = 100
    data_augmentation = True
    l1_reg = 0
    l2_reg = 1e-4

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    # Set all random seeds
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = '%s_exp%d.h5' % (filename, exp_id)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    # Get logger
    log_file = '%s_exp%d.log' % (filename, exp_id)
    log = logging.getLogger(filename)
    log.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(name)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    log.info(log_file)
    log.info(('BagNet GTSRB | seed: {}, init_learning_rate: {}, ' +
              'batch_size: {}, l2_reg: {}, l1_reg: {}, epochs: {}, ' +
              'data_augmentation: {}, subtract_pixel_mean: {}').format(
                  seed, 1e-3, batch_size, l2_reg, l1_reg, epochs,
                  data_augmentation, subtract_pixel_mean))

    log.info('Preparing data...')
    # trainloader, validloader, testloader = load_cifar10(batch_size,
    #                                                     data_dir='/data',
    #                                                     val_size=0.1,
    #                                                     augment=True,
    #                                                     shuffle=True,
    #                                                     seed=seed)
    trainloader, validloader, testloader = load_gtsrb_dataloader(
        '/data/', batch_size, num_workers=8)

    log.info('Building model...')
    # net = SmallBagNet(Bottleneck, [3, 4, 6, 3], strides=[2, 2, 2, 1],
    #                   kernel3=[1, 1, 0, 0], num_classes=43, patch_size=8,
    #                   patch_stride=8)
    # net = SmallBagNet(BottleneckNoBn, [2, 2, 2, 0], strides=[2, 2, 2, 1],
    #                   kernel3=[1, 1, 0, 0], num_classes=43, patch_size=8,
    #                   patch_stride=8)
    net = TinyBagNet(BottleneckNoBn, [2, 2, 2, 0], strides=[1, 2, 2, 1],
                     kernel3=[1, 1, 1, 0], num_classes=43, patch_size=8,
                     patch_stride=8)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=l2_reg)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [50, 70, 90], gamma=0.1)

    log.info(' epoch | loss  , acc    | val_loss, val_acc')
    best_acc = 0
    for epoch in range(epochs):
        lr_scheduler.step()
        best_acc = train(net, trainloader, validloader, criterion, optimizer,
                         epoch, device, log, save_best_only=True,
                         best_acc=best_acc, model_path=model_path)

    test_loss, test_acc = evaluate(net, testloader, criterion, device)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    main()
