import math
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
# from timm.utils import accuracy
from typing import Iterable, Optional
import utils.misc as misc
import utils.lr_sched as lr_sched
from sklearn.metrics import roc_auc_score
from pycm import *
import matplotlib.pyplot as plt
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None and args.log_tensorboard:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in samples.items()
        }

        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in args.in_domains
        }
        with torch.cuda.amp.autocast():
            outputs = model(input_dict)
            outputs = outputs['cls']
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and args.log_tensorboard:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

        if log_writer is not None and args.log_wandb:
            log_writer.update(
                {
                    'loss': loss_value_reduce,
                    'lr': max_lr,
                }
            )
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class, args, save_confusion_matrix=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task, exist_ok=True)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in images.items()
        }

        target = target.to(device, non_blocking=True)
        true_label = F.one_hot(target.to(torch.int64), num_classes=num_class)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
        }

        with torch.cuda.amp.autocast():
            output = model(input_dict)
            output = output['cls']
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _, prediction_decode = torch.max(prediction_softmax, 1)
            _, true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        # acc1, _ = accuracy(output, target, topk=(1, 2))
        metric_logger.update(loss=loss.item())

    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)

    if args.nb_classes == 2:
        auc_roc = roc_auc_score(true_label_decode_list, np.array(prediction_list)[:, 1])
    else:
        auc_roc = roc_auc_score(true_label_decode_list, np.array(prediction_list), multi_class='ovr', average='macro')
    metric_logger.synchronize_between_processes()
    
    print(f'Epoch {epoch} {mode} Sklearn Metrics - AUROC: {auc_roc:.4f}')

    if save_confusion_matrix:
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(os.path.join(task, f'confusion_matrix_{mode}.jpg'), dpi=600, bbox_inches='tight')

    metric_dict = {'epoch': epoch,'AUROC': auc_roc}

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_dict
    