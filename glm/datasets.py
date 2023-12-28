# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/01/11 21:01:51
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import torch
import random

import numpy as np
import pickle

from torch.utils.data import Dataset
from bisect import bisect_right


class LMDBDataset(Dataset):
    def __init__(self, path, process_fn):
        import lmdb
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.process_fn = process_fn
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = str(idx).encode('utf-8')
            row = pickle.loads(txn.get(key))
            return self.process_fn(row)


class BinaryDataset(Dataset):
    def __init__(self, path, process_fn, length_per_sample=64 + 1024 + 4096, dtype='int32', preload=False,
                 **kwargs):  # TODO ARGS
        assert length_per_sample is not None
        self.length_per_sample = length_per_sample
        self.dtype = np.dtype(dtype)
        self.process_fn = process_fn
        if preload:
            self.bin = np.fromfile(path, dtype=self.dtype).reshape(-1, length_per_sample)
        else:
            with open(path, 'r') as fid:
                nbytes = fid.seek(0, 2)
                flen = fid.tell() // self.dtype.itemsize
            self.bin = np.memmap(path, dtype=self.dtype, shape=(flen // length_per_sample, length_per_sample))

    def __len__(self):
        return self.bin.shape[0]

    def __getitem__(self, index):
        return self.process_fn(self.bin[index], index)


class TSVDataset(Dataset):
    def __init__(self, path, process_fn, with_heads=True, **kwargs):
        self.process_fn = process_fn
        with open(path, 'r') as fin:
            if with_heads:
                self.heads = fin.readline().split('\t')
            else:
                self.heads = None
            self.items = [line.split('\t') for line in fin]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.process_fn(self.items[index])


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence, weights):
        r, s = [], 0
        for i, e in enumerate(sequence):
            l = int(len(e) * weights[i])
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, weights=None, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if weights is None:
            self.weights = [1] * len(self.datasets)
        else:
            self.weights = weights
        self.cumulative_sizes = self.cumsum(self.datasets, self.weights)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]


class RandomMappingDataset(Dataset):
    '''
    Dataset wrapper to randomly mapping indices to original order.
    Will also enlarge the length
    '''
    def __init__(self, ds, scale=200, seed=None, **kwargs):
        self.wrapped_data = ds
        self.scale = scale
        self.seed = random.Random(seed).randint(0, 2**32-1) if seed is not None else 0

    def __len__(self):
        return len(self.wrapped_data) * self.scale

    def __getitem__(self, index):
        rng = random.Random(index)
        rng = np.random.RandomState(seed=[self.seed ^ rng.randint(0, 2**32-1) for _ in range(16)])
        index = rng.randint(len(self.wrapped_data))
        return self.wrapped_data[index]


class TransformingDataset(Dataset):
    '''
    Dataset wrapper to gradually change one to another during training
    ds1 -> ds2 in [start, end], calculate iteration based on consumed samples
    assume ds1 or ds2 is random mapping dataset
    '''
    def __init__(self, ds1, ds2, start, end, iteration, local_batch_size, if_print=False):
        self.ds1 = ds1
        self.ds2 = ds2
        self.start = start
        self.end = end
        self.init_iteration = iteration
        self.consumed_samples = 0
        self.local_batch_size = local_batch_size
        self.if_print = if_print
        if if_print:
            print(f'transforming [{start}, {end}), local-batch-size: {local_batch_size}')

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, index):
        iteration = self.init_iteration + (self.consumed_samples) / self.local_batch_size
        self.consumed_samples += 1
        if self.if_print and int((self.consumed_samples - 1) / self.local_batch_size) != \
                int((self.consumed_samples) / self.local_batch_size):
            print(f'[Rank {torch.distributed.get_rank()}] iteration: {int(iteration)}')

        ratio = 0
        if iteration >= self.end:
            ratio = 1
        elif self.start <= iteration < self.end:
            ratio = (iteration - self.start) / (self.end - self.start)

        rng = random.Random(index)
        rng = np.random.RandomState(seed=[rng.randint(0, 2**32-1) for _ in range(16)])
        if rng.random() < 1 - ratio:
            # print(f'[Rank {torch.distributed.get_rank()}] iteration: {iteration}, ratio: {ratio}, get ds1 {index} / {len(self.ds1)}')
            return self.ds1[index]
        else:
            # print(f'[Rank {torch.distributed.get_rank()}] iteration: {iteration}, ratio: {ratio}, get ds2 {index}')
            return self.ds2[index]


class BlockedRandomSplitDataset(Dataset):
    '''
    Dataset wrapper to access a subset of another dataset.
    Use block algorithm to reduce memory.
    In each block, using the `indices` items.
    '''
    def __init__(self, ds, indices, block_size,**kwargs):
        if type(indices) is not np.ndarray:
            indices = np.array(indices)
        indices = np.sort(indices)
        self.block_size = block_size
        self.wrapped_data = ds
        self.wrapped_data_len = len(ds)
        self.indices = indices
        self.len = len(indices) * (len(ds) // block_size) + np.sum(indices < (len(ds) % block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.wrapped_data[(index // len(self.indices)) * self.block_size + self.indices[index % len(self.indices)]]


class AggregatedDataset(Dataset):
    '''
    Dataset wrapper to aggregate multiple samples
    '''
    def __init__(self, ds, aggregated_sample_num, process_fn):
        self.wrapped_data = ds
        self.aggregated_sample_num = aggregated_sample_num
        self.process_fn = process_fn

    def __len__(self):
        return len(self.wrapped_data) // self.aggregated_sample_num

    def __getitem__(self, index):
        return self.process_fn([self.wrapped_data[index * self.aggregated_sample_num + offset]
                    for offset in range(self.aggregated_sample_num)])


class RandomGreedilyAggregatedDataset(Dataset):
    '''
    Random dataset aggregated dataset with greedy concat strategy
    '''
    def __init__(self, ds, max_seq_length, process_fn, seed=None):
        self.wrapped_data = ds
        self.max_seq_length = max_seq_length
        self.process_fn = process_fn
        self.seed = random.Random(seed).randint(0, 2**32-1) if seed is not None else 0

    def __len__(self):
        return len(self.wrapped_data)

    def __getitem__(self, index):
        rng = random.Random(index)
        rng = np.random.RandomState(seed=[self.seed ^ rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        items, length = [], 0

        while True:
            index = rng.randint(len(self.wrapped_data))
            item = self.wrapped_data[index]
            new_length = len(item[0]) + len(item[1]) + 2
            if length + new_length > self.max_seq_length:
                if length == 0:  # only one example, so we must append it then truncate
                    items.append(item)
                break
            length += new_length
            items.append(item)

        return self.process_fn(items)


def split_ds(ds, split=[.8,.2,.0], block_size = 10000, seed=1130):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
    """
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split, dtype=np.float32)
    split /= split.sum()

    assert block_size <= len(ds)

    start_idx = 0
    residual_idx = 0
    rtn_ds = [None]*len(split)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(np.array(range(block_size)))
    for i, f in enumerate(split):
        if f != 0:
            proportion = block_size*split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            rtn_ds[i] = BlockedRandomSplitDataset(ds, indices[range(start_idx, start_idx+max(split_, 1))], block_size)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds
