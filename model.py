#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn


def round_half_up(number):
    if number - math.floor(number) < 0.5:
        return math.floor(number)
    return math.ceil(number)


class TDC(nn.Module):
    def __init__(self, in_channels, in_channels_origin, out_channels, kernel_size, dropout_p):
        super().__init__()
        self.tdc = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=kernel_size, groups=in_channels, padding='same')
        self.pwc = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.resid_pwc = nn.Conv1d(in_channels=in_channels_origin, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.resid_bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, input_seq, origin_seq):
        # residual
        resid_conv_res = self.resid_pwc(origin_seq)
        resid_conv_res = self.resid_bn(resid_conv_res)
        # main
        conv_res = self.tdc(input_seq)
        conv_res = self.pwc(conv_res)
        conv_res += resid_conv_res
        conv_res = self.relu(conv_res)
        conv_res = self.dropout(conv_res)
        return conv_res


class QuartzNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_p, blocks=2):
        super().__init__()
        assert blocks > 0
        self.b_first  = TDC(in_channels,  in_channels, out_channels, kernel_size, dropout_p)
        self.b_second = TDC(out_channels, in_channels, out_channels, kernel_size, dropout_p)
    
    def forward(self, input_seq):
        block_res = self.b_first(input_seq, input_seq)
        block_res = self.b_second(block_res, input_seq)
        return block_res

class EouVadLSTM(nn.Module):
    def __init__(
        self, eou_window_size, n_mels, hidden_size, num_layers, batch_size, dropout, batch_first=True, classes_no=1, **_
    ):
        super().__init__()
        self.eou_window_size = eou_window_size
        self.n_mels = n_mels
        input_size = eou_window_size * n_mels
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.linear_eou = nn.Linear(hidden_size, classes_no)
        self.linear_vad = nn.Linear(hidden_size, classes_no)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = None

    def init_hidden(self, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )

    def forward(self, input_seq, h0=None, c0=None):  # pylint: disable=W0221
        if (h0 is not None) and (c0 is not None):
            output_seq, self.hidden = self.lstm(input_seq, (h0, c0))
        else:
            output_seq, self.hidden = self.lstm(input_seq)
        output_seq = self.dropout(output_seq)
        eou_predictions = self.linear_eou(output_seq)
        vad_predictions = self.linear_vad(output_seq)
        return eou_predictions, vad_predictions


class EouVadModelTDCLSTM(nn.Module):
    def __init__(
        self, eou_window_size, n_mels, hidden_size, num_layers, batch_size, dropout, batch_first=True, classes_no=1, **_
    ):
        super().__init__()
        self.eou_window_size = eou_window_size
        self.n_mels = n_mels
        
        self.conv_prologue = nn.Conv1d(in_channels=n_mels, out_channels=128, kernel_size=3, padding='same')
        self.qurtznet = QuartzNetBlock(128, 64, 7, dropout_p=0.2, blocks=2)
        self.conv_epilogue = nn.Conv1d(in_channels=64, out_channels=hidden_size, kernel_size=10)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.linear_eou = nn.Linear(hidden_size, classes_no)
        self.linear_vad = nn.Linear(hidden_size, classes_no)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = None

    def init_hidden(self, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )

    def forward(self, input_seq, h0=None, c0=None):  # pylint: disable=W0221
        bs_dim, seq_dim = input_seq.shape[0:2]
        input_seq = input_seq.reshape(-1, self.eou_window_size, self.n_mels)
        # n_mels as channels, convolution over time
        input_seq = input_seq.permute(0, 2, 1)

        conv_res = self.conv_prologue(input_seq)
        conv_res = self.relu(conv_res)
        conv_res = self.qurtznet(conv_res)
        conv_res = self.conv_epilogue(conv_res)  # -> [bch_s x seq_s, hid_s, 1]
        conv_res = self.relu(conv_res)
        conv_res = conv_res.reshape(bs_dim, seq_dim, -1)
    
        if (h0 is not None) and (c0 is not None):
            output_seq, self.hidden = self.lstm(conv_res, (h0, c0))
        else:
            output_seq, self.hidden = self.lstm(conv_res)
        output_seq = self.dropout(output_seq)
        eou_predictions = self.linear_eou(output_seq)
        vad_predictions = self.linear_vad(output_seq)
        return eou_predictions, vad_predictions


class EouVadModelInference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, h0=None, c0=None):  # pylint: disable=R0914, W0221
        if (h0 is not None) and (c0 is not None):
            y1, y2 = self.model(x, h0, c0)
        else:
            y1, y2 = self.model(x)
        y1 = self.sigmoid(y1)
        y2 = self.sigmoid(y2)
        return (y1, y2), self.model.hidden
