#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=E0401

import os
from collections.abc import Sequence

import json
import hydra
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset
import onnxruntime as rt

from model import EouVadModelTDCLSTM, EouVadModelInference, EouVadLSTM
import utils
from dataset import EouVadDataset


def calc_same_padding(kernel_size, stride, input_size):
    if isinstance(kernel_size, Sequence):
        kernel_size = kernel_size[0]

    if isinstance(stride, Sequence):
        stride = stride[0]

    if isinstance(input_size, Sequence):
        input_size = input_size[0]
    pad = ((stride - 1) * input_size - stride + kernel_size) / 2
    return int(pad)


class Exporter:
    def __init__(self, config, model, test_dataset: Dataset):
        self.config = config
        self.device = torch.device(self.config['cuda_device'] if torch.cuda.is_available() else "cpu")
        self.model = EouVadModelInference(model)
        self.model.to(self.device)
        self.model.eval()
        self.test_dataset = test_dataset
        # output path
        output_onnx = self.config.get('output_onnx', './eou.onnx')
        self.output_onnx = utils.create_abs_path(output_onnx)
        assert utils.is_file_creatable(self.output_onnx), "Output path can not ba accessed"
        output_json = self.config.get('output_json', './output.json')
        self.output_json = utils.create_abs_path(output_json)
        assert utils.is_file_creatable(self.output_json), "Output path can not ba accessed"
        # compare
        self.threshold = self.config.get('threshold', 1e-07)

    @staticmethod
    def replace_conv2d_with_same_padding(m: torch.nn.Module, input_size=10):
        if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
            if m.padding == "same":
                m.padding = calc_same_padding(
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    input_size=input_size
                )
    
    @staticmethod
    def get_dummpy_input(dataset: Dataset, device):
        dummy_input = dataset[0]
        dummy_tensor, eou_label, vad_label, _id, *_ = dummy_input
        dummy_tensor = dummy_tensor.unsqueeze(0).to(device)
        dummy_tensor = torch.flatten(dummy_tensor, start_dim=2)
        return dummy_tensor, eou_label, vad_label, _id

    @staticmethod
    def compare_outputs(output1, output2, threshold=1e-07):
        return torch.allclose(output1, output2, atol=threshold)

    def export(self):
        assert self.model
        assert self.output_onnx
        self.model.apply(lambda m: self.replace_conv2d_with_same_padding(m, 10))
        dummy_tensor, *_ = self.get_dummpy_input(self.test_dataset, self.device)
        h0, c0 = self.model.model.init_hidden(batch_size=1)  # pylint: disable=invalid-name
        torch.onnx.export(
            self.model,
            (dummy_tensor, h0.to(self.device), c0.to(self.device)),
            self.output_onnx,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input', 'h0', 'c0'],
            output_names=['eou', 'vad', 'hn', 'cn'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence'},
                'eou': {0: 'batch_size', 1: 'sequence'},
                'vad': {0: 'batch_size', 1: 'sequence'},
            },
            verbose=True,
        )
        return self.output_onnx

    @staticmethod
    def dump_json(json_path: str, data: dict):
        with open(json_path, "w") as w_file:
            json.dump(data, w_file, ensure_ascii=False)
            w_file.write('\n')

    def validate_export(self):
        onnx_model = rt.InferenceSession(self.config.output_onnx)
        # prepare inputs
        dummy_input = self.get_dummpy_input(self.test_dataset, self.device)
        dummy_tensor, eou_label, vad_label, _id = dummy_input
        in_h0, in_c0 = self.model.model.init_hidden(batch_size=1)
        # inference torch
        (eou_torch, vad_torch), _ = self.model(dummy_tensor, in_h0.to(self.device), in_c0.to(self.device))
        eou_torch = eou_torch.squeeze(2)
        vad_torch = vad_torch.squeeze(2)
        # inference onnx
        (eou_onnx, vad_onnx) = onnx_model.run(
            ['eou', 'vad'], {'input': dummy_tensor.cpu().numpy(), 'h0': in_h0.numpy(), 'c0': in_c0.numpy()}
        )
        eou_onnx = torch.from_numpy(eou_onnx).to(eou_torch.device).squeeze(2)
        vad_onnx = torch.from_numpy(vad_onnx).to(eou_torch.device).squeeze(2)
        # dump outputs
        data = {
            'id': _id,
            'eou_label': eou_label.cpu().detach().numpy().tolist(),
            'eou_torch': eou_torch.cpu().detach().numpy().tolist(),
            'eou_onnx': eou_onnx.cpu().detach().numpy().tolist(),
            'vad_label': vad_label.cpu().detach().numpy().tolist(),
            'vad_torch': vad_torch.cpu().detach().numpy().tolist(),
            'vad_onnx': vad_onnx.cpu().detach().numpy().tolist(),
        }
        self.dump_json(self.output_json, data)
        # validate
        o_close_eou = self.compare_outputs(eou_torch, eou_onnx, threshold=self.threshold)
        if not o_close_eou:
            raise Exception('EOU output of model and onnx model were not close enough.' +
                            f'Watch {self.output_json} for details')
        o_close_vad = self.compare_outputs(vad_torch, vad_onnx, threshold=self.threshold)
        if not o_close_vad:
            raise Exception('EOU output of model and onnx model were not close enough.' +
                            f'Watch {self.output_json} for details')


@hydra.main(config_path="configs", config_name="export")
def main(config: DictConfig):
    # create model
    recipe = config.recipe
    if recipe.model_type == "LSTM":
        model_cls = EouVadLSTM
    else:
        model_cls = EouVadModelTDCLSTM
    model = model_cls(**recipe, batch_first=True, classes_no=1)
    utils.load_checkpoint(config.input_checkpoint, model)

    # create test dataloader
    test_dataset = EouVadDataset(**config.test_ds, max_size=1)

    exporter_config = config.exporter
    exporter = Exporter(exporter_config, model, test_dataset)
    onnx_path = exporter.export()
    exporter.validate_export()

    print(f"ONNX model was successfully exported to '{os.path.abspath(onnx_path)}'")


if __name__ == '__main__':
    # pylint: disable = no-value-for-parameter
    main()
