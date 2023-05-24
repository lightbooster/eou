# -*- coding: utf-8 -*-
import os
import random
import itertools
from abc import abstractclassmethod
from typing import List
from random import choices, sample
from importlib import import_module

import numpy as np
import pandas as pd
import audioaug
from omegaconf import ListConfig
from hydra.utils import to_absolute_path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


SPOTTER_WORDS = ['салют', 'сбер', 'джой', 'афина', 'алиса', 'сири', 'помощник']
spotter_words_with_privet = [f"привет {spotter}" for spotter in SPOTTER_WORDS]
SPOTTER_WORDS += spotter_words_with_privet


def get_object_from_string(object_name: str, obj_type='obj') -> object:
    if obj_type not in ['obj', 'enm']:
        raise RuntimeError("Object type must be 'obj' for class object or 'enm' for Enum value")
    try:
        if obj_type == 'enm':
            class_name, enum_value = object_name.rsplit('.', 2)
        else:
            class_name = object_name

        module = import_module('audioaug')
        class_obj = getattr(module, class_name)

        if obj_type == 'obj':
            return class_obj
        elif obj_type == 'enm':
            return getattr(class_obj, enum_value)

    except (ImportError, AttributeError) as e:
        raise ImportError(object_name)


def instance_from_string(object_name: str, kwargs: dict) -> object:
    class_name = get_object_from_string(object_name)
    kwargs = dict(kwargs)

    for key, value in kwargs.items():
        if isinstance(value, str) and (value.startswith('obj:') or value.startswith('enm:')):
            # replace string with object
            kwargs[key] = get_object_from_string(object_name=value[4:], obj_type=value[:3])

    return class_name(**kwargs)


def eou_label_to_array(label, length, batch_size):
    size = int((label - 1) / batch_size)
    if length < size:
        return np.zeros([length])
    a = np.zeros([size], dtype=np.int32)
    b = np.ones([length - size], dtype=np.int32)
    return np.concatenate((a, b))


def vad_label_to_array(label, length, batch_size, sample_rate, melspec_ms):
    a = np.zeros(length)
    q = 1000 / melspec_ms
    for start, end in label:
        start, end = start / sample_rate, end / sample_rate
        start, end = start * q // batch_size, end * q // batch_size
        start, end = int(start), int(end)
        a[start : end] = 1
    return a


def pad_collate(batch_val):
    (x_vals, y1_vals, y2_vals, ids, words_timing) = zip(*batch_val)
    x_lens = [len(x) for x in x_vals]
    y1_lens = [len(y) for y in y1_vals]
    y2_lens = [len(y) for y in y2_vals]

    xx_pad = pad_sequence(x_vals, batch_first=True, padding_value=0)
    yy1_pad = pad_sequence(y1_vals, batch_first=True, padding_value=0)
    yy2_pad = pad_sequence(y2_vals, batch_first=True, padding_value=0)

    return xx_pad, (yy1_pad, yy2_pad), x_lens, (y1_lens, y2_lens), ids, words_timing


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


class EouVadDataset(Dataset):
    class Augmentor:
            def __init__(self, augmentations: list, probabilities: list) -> None:
                self.augmentations = augmentations
                self.probabilities = probabilities
                assert len(self.probabilities) == len(self.augmentations)

            def perturbate(self, wav, info):
                aug = choices(self.augmentations, self.probabilities)[0]
                wav, info = audioaug.apply_augmentation(wav, info, aug, choose_channel_index=0)
                return wav, info

            @classmethod
            def from_config(cls, config) -> 'EouDataset.Augmentor':
                assert len(config.probabilities) == len(config.augmentations)
                all_augs = []
                for augmentation_list in config.augmentations:
                    augs = []
                    for aug in augmentation_list:
                        aug_inst = instance_from_string(aug['name'], aug['kwargs'])
                        augs.append(aug_inst)
                    all_augs.append(augs)
                return cls(all_augs, config.probabilities)
    def __init__(
        self,
        manifest_path,
        *_,
        weights=None,
        sample_rate=8_000,
        n_mels=64,
        melspec_ms=10,
        eou_window_size=10,
        min_silence_len=0.0,
        remove_empty=False,
        pad_target_seconds=0.0,
        pad_unigram_seconds=0.0,
        feature_extraction_params=None,
        max_size=-1,
        unigrams=None,
        augmentor=None,
        random_state=0,
        **__,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.melspec_ms = melspec_ms
        self.eou_window_size = eou_window_size
        self.pad_target_seconds = pad_target_seconds
        self.pad_unigram_seconds = pad_unigram_seconds
        # unigrams
        self.unigrams = unigrams
        if self.unigrams is None:
            self.unigrams = []
        unigrams_texts = self.unigrams.copy()
        for unigram in self.unigrams:
            unigrams_texts += [f"{spotter} {unigram}" for spotter in SPOTTER_WORDS]
        self.unigrams = {unigram_text: True for unigram_text in unigrams_texts}
        print("Unigrams: ", self.unigrams.keys())

        random.seed(random_state)
        if isinstance(manifest_path, (ListConfig, List)):
            assert weights is None or (len(weights) == len(manifest_path)), \
                "Lists of manifests and weights should have same lengths"
            if weights is not None:
                weights = np.array(weights)
            else:
                weights = np.ones(shape=len(manifest_path))
            assert np.any((weights >= 0.0) & (weights <= 1.0)), \
                f"Weight value should be between 0 and 1, actual: {weights}"
            data_list = []
            for man, weight in zip(manifest_path, weights):
                man = to_absolute_path(man)
                data = self.read_data(
                    man, min_silence_len=min_silence_len, remove_empty=remove_empty, random_state=random_state
                )
                data_size = int(len(data) * weight)
                data = random.sample(data, data_size)
                data_list.append(data)
            self.data = list(itertools.chain.from_iterable(data_list))
            if max_size > 0:
                max_size = min(max_size, len(self.data))
                self.data = random.sample(self.data, max_size)
        else:
            manifest_path = to_absolute_path(manifest_path)
            self.data = self.read_data(
                manifest_path, min_silence_len=min_silence_len, remove_empty=remove_empty, max_size=max_size, random_state=random_state
            )
        if augmentor is not None:
            self.augmentor = EouVadDataset.Augmentor.from_config(augmentor)
        else:
            self.augmentor = None
        self.extraction_params = audioaug.FeatureParams(**feature_extraction_params)


    @classmethod
    def prepare_tensor(cls, features, n_mels=64, eou_window_size=10):
        t_64 = np.reshape(features, (-1, n_mels))
        t_25_64 = [torch.from_numpy(b) for b in batch(t_64, eou_window_size)]
        stub = [torch.empty([eou_window_size, n_mels])]
        t_25_64 = stub + t_25_64
        padded = pad_sequence(t_25_64, batch_first=True, padding_value=0)
        return padded[1:]

    @classmethod
    def load_tensor(cls, path, n_mels=64, eou_window_size=10):
        features = np.fromfile(path, dtype=np.float32)
        return cls.prepare_tensor(features, n_mels=n_mels, eou_window_size=eou_window_size)

    def load_and_extract_waveform(self, path, intervals):
        wav = None
        info = None
        for i in range(4):
            try:
                wav, info = audioaug.read_waveform(path)
                break
            except Exception as e:
                if i == 3:
                    raise e
        info_ext = audioaug.AudioInfoExt.copy(info)
        info_ext.intervals = intervals
        if self.augmentor is not None:
            wav, info = self.augmentor.perturbate(wav, info_ext)
        features = audioaug.extract_features(wav, info, self.extraction_params)
        return self.prepare_tensor(features)

    @staticmethod
    def load_intervals(path):
        if path is None or not os.path.exists(path):
            return None
        intervals = np.fromfile(path, dtype=np.float32)
        intervals = intervals.reshape([-1, 2])
        return intervals

    def read_data(self, path, min_silence_len=0.0, remove_empty=False, max_size=-1, random_state=0):
        df = pd.read_csv(path, sep='\t', na_filter=False)
        if max_size > 0:
            max_size = min(max_size, len(df))
            df = df.sample(max_size, random_state=random_state)
        data = []
        min_len = 2
        for _, row in df.iterrows():
            hyp = row.get('hyp', '')
            if row.get('diff', 0.0) >= min_silence_len and len(hyp) >= min_len:
                pad_value = self.pad_target_seconds
                if hyp in self.unigrams:
                    pad_value = self.pad_unigram_seconds
                data.append(
                    {
                        'id': row['id'],
                        'eou_label': int((row['eou'] + pad_value) * (1000 / self.melspec_ms)),
                        'vad_label': os.path.join(os.path.dirname(path), row['vad']),
                        'path': os.path.join(os.path.dirname(path), row['path']),
                        'intervals': os.path.join(os.path.dirname(path), row['interval_path']),
                    }
                )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        intervals = self.load_intervals(self.data[idx]['intervals'])
        if self.data[idx]['path'].endswith('.wav'):
            tensor = self.load_and_extract_waveform(self.data[idx]['path'], intervals)
        else:
            tensor = self.load_tensor(self.data[idx]['path'], n_mels=self.n_mels, eou_window_size=self.eou_window_size)
        eou_label = torch.from_numpy(self.load_eou_label(self.data[idx]['eou_label'], tensor.shape[0]))
        vad_label = torch.from_numpy(self.load_vad_label(self.data[idx]['vad_label'], tensor.shape[0]))
        id = self.data[idx]['id']
        return tensor, eou_label, vad_label, id, intervals

    def load_eou_label(self, label, length):
        return eou_label_to_array(label, length, self.eou_window_size)
    
    def load_vad_label(self, label, length):
        label = np.fromfile(label, dtype=np.int32).reshape(-1, 2)
        return vad_label_to_array(label, length, self.eou_window_size, self.sample_rate, self.melspec_ms)

    def get_dataloader(self, batch_size, n_workers, prefetch_factor, shuffle=True, **_):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=n_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=pad_collate,
            shuffle=shuffle,
        )
