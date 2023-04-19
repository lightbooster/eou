# -*- coding: utf-8 -*-
import os
import pathlib
from typing import List
from collections import namedtuple

import numpy as np
import torch

TRAIN_VAD = True

def save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path):
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"Saving model and optimizer state at epoch {epoch} to '{checkpoint_path}'")
    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(save_dict, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
    print(f"Loaded checkpoint '{checkpoint_path}' from epoch {epoch}")
    return epoch


class LossCoefScheduler():
    def __init__(self, epochs: int, start: float, end: float, *_, **__) -> None:
        self.epochs = epochs - 1
        assert self.epochs > 0
        self.start = start
        self.end = end
    
    @staticmethod
    def smooth_func(x):
        return 0.5 * np.cos(np.pi * x) + 0.5

    def get_coef(self, epoch):
        if epoch >= self.epochs:
            epoch = self.epochs
        x = epoch / self.epochs
        smooth_value = self.smooth_func(x)
        diff = self.start - self.end
        return self.end + smooth_value * (diff)


class RunningCollector:
    """
    Collects and agragate values during running model on a dataset
    """

    RunningValues = namedtuple('Values', 'loss loss_eou loss_vad eou_probs vad_probs eou_preds vad_preds eou_labels vad_labels intervals')

    def __init__(self, device, dataset_size: int) -> None:
        self.device = device
        self.dataset_size = dataset_size
        self.loss: float = 0.0
        self.loss_eou: float = 0.0
        self.loss_vad: float = 0.0
        self.eou_probs: List[torch.Tensor] = []
        self.vad_probs: List[torch.Tensor] = []
        self.eou_preds: torch.Tensor = torch.zeros(0, device=self.device, dtype=torch.int)
        self.vad_preds: torch.Tensor = torch.zeros(0, device=self.device, dtype=torch.int)
        self.eou_labels: torch.Tensor = torch.zeros(0, device=self.device, dtype=torch.int)
        self.vad_labels: torch.Tensor = torch.zeros(0, device=self.device, dtype=torch.int)
        self.intervals: list = []

    def add(
        self,
        loss: float,
        loss_eou: float,
        loss_vad: float,
        probs,
        preds,
        labels,
        intervals: list,
    ):
        (eou_probs, vad_probs) = probs
        (eou_preds, vad_preds) = preds
        (eou_labels, vad_labels) = labels
        self.loss += loss
        self.loss_eou += loss_eou
        self.loss_vad += loss_vad
        self.eou_probs.append(eou_probs)
        self.vad_probs.append(vad_probs)

        self.vad_preds = torch.cat((self.vad_preds, vad_preds.flatten()))
        self.eou_preds = torch.cat((self.eou_preds, eou_preds.flatten()))

        self.vad_labels = torch.cat((self.vad_labels, vad_labels.flatten()))
        self.eou_labels = torch.cat((self.eou_labels, eou_labels.flatten()))

        self.intervals += intervals

    def get_values(self, get_numpy=True):
        max_probs_len = max([t.shape[-1] for t in self.eou_probs])
        running_eou_probs = [torch.cat([t, t.new_zeros((t.shape[0], max_probs_len - t.shape[-1]))], -1) for t in self.eou_probs]
        running_eou_probs = torch.cat(running_eou_probs, dim=0)
        running_vad_probs = [torch.cat([t, t.new_zeros((t.shape[0], max_probs_len - t.shape[-1]))], -1) for t in self.vad_probs]
        running_vad_probs = torch.cat(running_vad_probs, dim=0)
        rprobs_vad = running_vad_probs.cpu().detach()
        rprobs_eou = running_eou_probs.cpu().detach()
        rpreds_vad = self.vad_preds.cpu().detach()
        rpreds_eou = self.eou_preds.cpu().detach()
        rlabls_vad = self.vad_labels.cpu().detach()
        rlabls_eou = self.eou_labels.cpu().detach()
        if get_numpy:
            rprobs_vad = rprobs_vad.numpy()
            rprobs_eou = rprobs_eou.numpy()
            rpreds_vad = rpreds_vad.numpy()
            rpreds_eou = rpreds_eou.numpy()
            rlabls_vad = rlabls_vad.numpy()
            rlabls_eou = rlabls_eou.numpy()
        values = self.RunningValues(
            self.loss / self.dataset_size,
            self.loss_eou / self.dataset_size,
            self.loss_vad / self.dataset_size,
            rprobs_eou, rprobs_vad,
            rpreds_eou, rpreds_vad,
            rlabls_eou, rlabls_vad,
            self.intervals,
        )
        return values


def __consistent_args(input_val, condition, indices):
    assert len(input_val.shape) == 2, 'only works for batch x dim tensors along the dim axis'
    mask = condition(input_val).float() * indices.unsqueeze(0).expand_as(input_val)
    return torch.argmax(mask, dim=1)


def consistent_find_leftmost(input_val, condition):
    indices = torch.arange(input_val.size(1), 0, -1, dtype=torch.float, device=input_val.device)
    return __consistent_args(input_val, condition, indices)


def consistent_find_rightmost(input_val, condition):
    indices = torch.arange(0, input_val.size(1), 1, dtype=torch.float, device=input_val.device)
    return __consistent_args(input_val, condition, indices)


def left_argmax(labels, thr=0.5):
    res = torch.empty(labels.shape[0], device=labels.device, dtype=torch.int)
    for i in range(labels.shape[0]):
        res[i] = consistent_find_leftmost(labels[i].unsqueeze(0), lambda x: x > thr)
    return res


def build_seq_mask(lens, outputs):
    seq_lens = torch.tensor(lens, device=outputs.device).unsqueeze(-1)  # pylint: disable=E1102
    max_len = outputs.shape[1]
    range_tensor = torch.arange(max_len, device=outputs.device).unsqueeze(0)
    range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.size(1))
    mask_tensor = range_tensor < seq_lens
    return mask_tensor


def build_fp_loss_mask(labels, mask, probs, fp_prob_threshold=0.99, max_value=10.0):
    false_positive = (probs > fp_prob_threshold) * (labels < 0.5)
    multiplier = max_value / (1 - fp_prob_threshold)
    diff = multiplier * (probs - fp_prob_threshold)
    z_mask = diff * false_positive
    z_mask = z_mask * z_mask
    z_mask += 1.0
    limit = torch.ones(z_mask.shape, dtype=torch.float32).to(mask.device)
    limit = max_value * limit
    z_mask = torch.where((z_mask > max_value), limit, z_mask)
    return z_mask * mask


def build_area_loss_mask(labels, mask, quad_area=-1, quad_left=False, quad_right=False, max_value=10.0):
    if quad_area < 0:
        return mask
    pos = left_argmax(labels, 0.95).unsqueeze(-1)
    z_mask = torch.arange(labels.shape[1], dtype=torch.int32).to(mask.device).expand(labels.shape[0], labels.shape[1])
    z_mask = z_mask - pos
    ones = torch.ones(z_mask.shape, dtype=torch.int32).to(mask.device)
    left_border = -quad_area
    right_border = quad_area
    z_mask = torch.where((z_mask >= left_border) & (z_mask <= right_border), ones, z_mask)
    if quad_left:
        z_mask = torch.where((z_mask > 0), ones, z_mask)
    if quad_right:
        z_mask = torch.where((z_mask < 0), ones, z_mask)
    z_mask = z_mask * z_mask
    limit = torch.ones(z_mask.shape, dtype=torch.int32).to(mask.device)
    limit = max_value * limit
    z_mask = torch.where((z_mask > max_value), limit, z_mask)
    return z_mask * mask


def get_loss_mask_builder(
    loss_type='AREA', max_value=10.0, quad_area=-1, quad_left=False, quad_right=False, quad_fp_prob_threshold=0.99
):
    def quad_loss(labels, mask, probs):
        if loss_type == 'AREA':
            return build_area_loss_mask(labels, mask, quad_area, quad_left, quad_right, max_value)
        elif loss_type == 'FP':
            return build_fp_loss_mask(labels, mask, probs, quad_fp_prob_threshold, max_value)
        else:
            return mask

    if loss_type not in ('AREA', 'FP'):
        return None
    return quad_loss


def is_file_creatable(path: str) -> bool:
    dirname = os.path.dirname(path) or os.getcwd()
    return os.access(dirname, os.W_OK)


def create_abs_path(path: str) -> str:
    if os.path.isabs(path):
        dir_path = os.path.dirname(path)
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    else:
        path = os.path.abspath(path)
    return path
