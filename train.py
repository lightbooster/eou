# -*- coding: utf-8 -*-

import os
import time
import copy

import hydra
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np

import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import utils
import metrics
from dataset import EouVadDataset
from warmup_scheduler import GradualWarmupScheduler
from model import EouVadModelTDCLSTM, EouVadLSTM
from export import Exporter


TRAIN_VAD = False
LOG_FILE_NAME = "vad_log_new04.txt"


class Learner:
    def __init__(self, config, model, train_dataloader: DataLoader, valid_dataloader: DataLoader):
        self.config = config
        self.debug = self.config.get('debug', False)
        self.device = torch.device(self.config['cuda_device'] if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.dataloaders = {}
        self.batch_size = self.config['batch_size']
        self.dataloaders = {
            'train': train_dataloader,
            'validate': valid_dataloader,
        }
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        quad_loss_config = self.config.get('quad_loss', None)
        if quad_loss_config is not None:
            self.loss_mask_builder = utils.get_loss_mask_builder(**quad_loss_config)
        else:
            self.loss_mask_builder = None

        show_server_like_metrics = self.config.get('show_server_like_metrics', False)
        server_eou_hp = self.config.get('server_eou_hp', None)
        if show_server_like_metrics:
            assert server_eou_hp is not None, "Describe server EOU hyperparams to show server-like metrics"

        if not self.debug:
            self.summary_writer = SummaryWriter(log_dir='.')
            output_checkpoint = self.config.get('output_checkpoint', './checkpoint.pt')
            self.output_checkpoint = utils.create_abs_path(output_checkpoint)
            assert utils.is_file_creatable(self.output_checkpoint), "Output path can not ba accessed"
        else:
            self.output_checkpoint = ""

    def train(self):
        # pylint: disable = R0912 (branches number)
        assert self.debug or self.output_checkpoint, "Output path is absent in a not debug run"
        optimizer = Adam(
            self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config.get('weight_decay', 0.0)
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config['opt_step'], gamma=self.config['opt_gamma'])
        if self.config.get('enable_cos_lr', False):
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config['epochs_scheduler'], eta_min=self.config['opt_gamma']
            )
        if self.config.get('enable_warmup', False):
            scheduler = GradualWarmupScheduler(
                optimizer, multiplier=1, total_epoch=self.config['epochs'] // 10, after_scheduler=scheduler
            )
        loss_coef_scheduler = utils.LossCoefScheduler(**self.config['loss_coef_scheduler'])

        # load checkpoint
        cur_epoch = 1
        checkpoint_path = self.config.get('pretrained_checkpoint', '')
        if checkpoint_path:
            if self.config.get('continue_from_checkpoint', False):
                cur_epoch = utils.load_checkpoint(checkpoint_path, self.model, optimizer, scheduler)
            else:
                _ = utils.load_checkpoint(checkpoint_path, self.model)

        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 100
        best_overall_loss = 100
        best_epoch = cur_epoch
        sigmoid = nn.Sigmoid()
        try:
            for epoch in range(cur_epoch, self.config['epochs'] + 1):
                cur_epoch = epoch
                loss_coef_eou = loss_coef_scheduler.get_coef(epoch)
                if loss_coef_eou > 1.0:
                    loss_coef_vad = 0.0
                elif loss_coef_eou < 0.0:
                    loss_coef_vad = 1.0
                else:
                    loss_coef_vad = 1.0 - loss_coef_eou
                print(f'Epoch {epoch}/{self.config["epochs"]}')
                for phase in ['train', 'validate']:
                    if phase == 'validate' and self.debug:
                        continue
                    if phase == 'train':
                        self.model.hidden = self.model.init_hidden()
                        self.model.train()
                        cur_step_lr = scheduler.get_last_lr()[-1]
                    else:
                        self.model.eval()

                    dataset_size = len(self.dataloaders[phase].dataset)
                    running_collector = utils.RunningCollector(device=self.device, dataset_size=dataset_size)
                    for inputs, (eou_labels, vad_labels), _, (labels_len, _), _, intervals in tqdm(self.dataloaders[phase]):
                        inputs = torch.flatten(inputs, start_dim=2).to(self.device)
                        eou_labels = eou_labels.float().to(self.device)
                        vad_labels = vad_labels.float().to(self.device)
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            eou_outputs, vad_outputs = self.model(inputs)
                            eou_outputs = eou_outputs.squeeze(2)
                            vad_outputs = vad_outputs.squeeze(2)

                            seq_mask = utils.build_seq_mask(labels_len, eou_outputs)
                            eou_probs = seq_mask * sigmoid(eou_outputs)
                            vad_probs = seq_mask * sigmoid(vad_outputs)
                            
                            loss_eou = (self.criterion(eou_outputs, eou_labels) * seq_mask).sum()
                            loss_vad = (self.criterion(vad_outputs, vad_labels) * seq_mask).sum()
                            loss = 2 * (loss_coef_eou * loss_eou + loss_coef_vad * loss_vad)

                            if phase == "train":
                                loss.backward()
                                optimizer.step()
                        
                        vad_preds = (vad_probs > 0.5).float()
                        eou_preds = utils.left_argmax(eou_probs)
                        eou_labels = utils.left_argmax(eou_labels)
                        running_collector.add(loss.item(), loss_eou.item(), loss_vad.item(),
                                             (eou_probs, vad_probs), (eou_preds, vad_preds), (eou_labels, vad_labels),
                                              intervals)

                    if phase == 'train' and epoch < self.config['epochs_scheduler']:
                        if self.config['enable_warmup']:
                            scheduler.step(epoch)
                        else:
                            scheduler.step()

                    # Metrics
                    running_values = running_collector.get_values(get_numpy=True)

                    eou_metrics = metrics.calculate_metrics(
                        running_values.eou_preds,
                        running_values.eou_labels,
                        running_values.intervals,
                        eou_step_seconds=(1.0 / self.model.eou_window_size),
                    )
                    log_message = metrics.create_metrics_message(
                        phase, running_values.loss, eou_metrics
                    )
                    print(log_message)

                    show_server_like_metrics = self.config.get('show_server_like_metrics', False)
                    if show_server_like_metrics:
                        server_preds = metrics.create_server_predictions(
                            running_values.eou_probs, **self.config.server_eou_hp
                        )
                        server_metrics = metrics.calculate_metrics(
                            server_preds,
                            running_values.eou_labels,
                            running_values.intervals,
                            eou_step_seconds=(1.0 / self.model.eou_window_size),
                        )
                        log_message = metrics.create_metrics_message(
                            phase, None, server_metrics, print_loss=False
                        )
                        print('Server EOU simulation:')
                        print(log_message)

                        # Tensorboard
                        if phase == 'validate' and not self.debug:
                            self.summary_writer.add_scalar("Loss_valid/loss", running_values.loss, epoch)
                            self.summary_writer.add_scalar("Loss_valid/loss_eou", running_values.loss_eou, epoch)
                            self.summary_writer.add_scalar("Loss_valid/loss_vad", running_values.loss_vad, epoch)

                            self.summary_writer.add_scalar("EouDistance/Q1", eou_metrics.p1, epoch)
                            self.summary_writer.add_scalar("EouDistance/Q5", eou_metrics.p5, epoch)
                            self.summary_writer.add_scalar("EouDistance/Q50", eou_metrics.median, epoch)
                            self.summary_writer.add_scalar("EouDistance/Q95", eou_metrics.p95, epoch)
                            self.summary_writer.add_scalar("EouDistance/Q99", eou_metrics.p99, epoch)
                            self.summary_writer.add_scalar("EouDistance/Avg", eou_metrics.avg, epoch)

                            self.summary_writer.add_scalar("Activations/True", eou_metrics.activations.true, epoch)
                            self.summary_writer.add_scalar("Activations/False", eou_metrics.activations.false, epoch)
                            self.summary_writer.add_scalar("Activations/Rate", eou_metrics.activations.rate, epoch)
                            self.summary_writer.add_scalar("AsrMetrics/WER", eou_metrics.wer, epoch)
                            self.summary_writer.add_scalar("AsrMetrics/SER", eou_metrics.ser, epoch)

                            if show_server_like_metrics:
                                self.summary_writer.add_scalar("ServerLike/Q1", server_metrics.p1, epoch)
                                self.summary_writer.add_scalar("ServerLike/Q5", server_metrics.p5, epoch)

                                self.summary_writer.add_scalar("ServerLike/Q50", server_metrics.median, epoch)
                                self.summary_writer.add_scalar("ServerLike/Q95", server_metrics.p95, epoch)
                                self.summary_writer.add_scalar("ServerLike/Q99", server_metrics.p99, epoch)
                                self.summary_writer.add_scalar("ServerLike/Avg", server_metrics.avg, epoch)

                                self.summary_writer.add_scalar(
                                    "ServerLike/ActivationsTrue", server_metrics.activations.true, epoch
                                )
                                self.summary_writer.add_scalar(
                                    "ServerLike/ActivationsFalse", server_metrics.activations.false, epoch
                                )
                                self.summary_writer.add_scalar(
                                    "ServerLike/ActivationsRate", server_metrics.activations.rate, epoch
                                )
                                self.summary_writer.add_scalar("ServerLike/WER", server_metrics.wer, epoch)
                                self.summary_writer.add_scalar("ServerLike/SER", server_metrics.ser, epoch)
                        elif phase == 'train' and not self.debug:
                            self.summary_writer.add_scalar("Loss_train/loss", running_values.loss, epoch)
                            self.summary_writer.add_scalar("Loss_train/loss_eou", running_values.loss_eou, epoch)
                            self.summary_writer.add_scalar("Loss_train/loss_vad", running_values.loss_vad, epoch)
                            self.summary_writer.add_scalar("LR/value", cur_step_lr, epoch)
                            self.summary_writer.add_scalar("LossCoef/eou", loss_coef_eou, epoch)
                            self.summary_writer.add_scalar("LossCoef/vad", loss_coef_vad, epoch)

                        
                        f1_vad = f1_score(running_values.vad_preds.flatten(), running_values.vad_labels.flatten().astype(np.int32), average='macro')
                        print(f'VAD F1Score: {f1_vad}\n')
                        if phase == 'validate':
                            cs_vad = classification_report(running_values.vad_labels.flatten().astype(np.int32),
                                                        running_values.vad_preds.flatten().astype(np.int32))
                            print(cs_vad)

                        # Tensorboard
                        if phase == 'validate' and not self.debug:
                            self.summary_writer.add_scalar("F1ScoreVAD", f1_vad, epoch)

                    if phase == 'validate' and running_values.loss < best_loss:
                        best_loss = running_values.loss
                        best_overall_loss = running_values.loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        best_epoch = cur_epoch

        except KeyboardInterrupt:
            pass

        time_elapsed = time.time() - since
        print(
            f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.'
            + f' Best model loss: {best_loss:.6f} Actual loss: {best_overall_loss:.6f}'
        )

        self.model.load_state_dict(best_model_wts)
        self.model.eval()

        if not self.debug:
            utils.save_checkpoint(best_epoch, self.model, optimizer, scheduler, self.output_checkpoint)

        return self.model, best_loss


@hydra.main(config_path="configs", config_name="train")
def main(config: DictConfig):
    print('cur_dir', os.path.abspath('.'))
    # create model
    recipe = config.recipe
    if recipe.model_type == "LSTM":
        model_cls = EouVadLSTM
    else:
        model_cls = EouVadModelTDCLSTM
    model = model_cls(**recipe, batch_first=True, classes_no=1)

    # create dataloaders
    max_size = 2 if config.debug else -1
    dataset_cls = EouVadDataset
    train_dataset = dataset_cls(**config.train_ds, max_size=max_size)
    valid_dataset = dataset_cls(**config.valid_ds, max_size=max_size)
    train_dataloader = train_dataset.get_dataloader(**config.train_ds)
    valid_dataloader = valid_dataset.get_dataloader(**config.valid_ds)

    # fit
    learner = Learner(config.learner, model, train_dataloader, valid_dataloader)
    model, _ = learner.train()

    # export to onnx
    exporter_config = config.get('exporter', None)
    if exporter_config is not None:
        print('Trained model will be exported to ONNX')
        exporter = Exporter(exporter_config, model, valid_dataloader.dataset)
        onnx_path = exporter.export()
        exporter.validate_export()
        print(f"ONNX model was successfully exported to '{onnx_path}'")


if __name__ == '__main__':
    # pylint: disable = no-value-for-parameter
    main()
