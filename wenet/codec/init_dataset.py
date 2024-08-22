from functools import partial
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from wenet.dataset import processor
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)


def padding(data):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)

    lengths = [
        min(len(obj['acoustic']), len(obj['semantic'])) for obj in sample
    ]
    acoustics = [
        torch.tensor(obj['acoustic'][:lengths[i]], dtype=torch.int)
        for i, obj in enumerate(sample)
    ]
    semantics = [
        torch.tensor(obj['semantic'][:lengths[i]], dtype=torch.int)
        for i, obj in enumerate(sample)
    ]

    padded_acoustics = pad_sequence(acoustics,
                                    batch_first=True,
                                    padding_value=0)
    padded_semantics = pad_sequence(semantics,
                                    batch_first=True,
                                    padding_value=0)
    lengths = torch.tensor(lengths)
    batch = {
        'acoustics': padded_acoustics,
        'semantics': padded_semantics,
        'lengths': lengths,
    }
    return batch


def filter_fn(example, semantic_min_tokens, semantic_max_tokens,
              acoustic_min_tokens, acoustic_max_tokens):
    acoustic_tokens = example['acoustic']
    semantic_tokens = example['semantic']

    lens_a = len(acoustic_tokens)
    lens_s = len(semantic_tokens)

    if lens_a >= semantic_min_tokens and lens_a <= semantic_max_tokens:
        return True
    if lens_s >= acoustic_min_tokens and lens_s <= acoustic_max_tokens:
        return True
    return False


def Dataset(data_type, data_list_file, conf=None, partition=True):
    """ Construct dataset from arguments for ssl model

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            partition(bool): whether to do data partition in terms of rank
    """
    """
        data format:
            {"semantic_token:" [], "acoustic_token": []}
    """
    assert conf is not None
    assert data_type in ['raw', 'shard']
    # cycle dataset
    cycle = conf.get('cycle', 1)
    # stage1 shuffle: source
    list_shuffle = conf.get('list_shuffle', True)

    list_shuffle_size = sys.maxsize
    if list_shuffle:
        list_shuffle_conf = conf.get('list_shuffle_conf', {})
        list_shuffle_size = list_shuffle_conf.get('shuffle_size',
                                                  list_shuffle_size)
    if data_type == 'raw':
        dataset = WenetRawDatasetSource(data_list_file,
                                        partition=partition,
                                        shuffle=list_shuffle,
                                        shuffle_size=list_shuffle_size,
                                        cycle=cycle)
        dataset = dataset.map(processor.parse_json)
    else:
        dataset = WenetTarShardDatasetSource(data_list_file,
                                             partition=partition,
                                             shuffle=list_shuffle,
                                             shuffle_size=list_shuffle_size,
                                             cycle=cycle)
    filter_conf = conf.get('filter_conf', {})
    dataset = dataset.filter(partial(filter_fn, **filter_conf))

    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
                               key_func=processor.sort_by_feats)

    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    assert batch_type in ['static', 'bucket', 'dynamic']
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(batch_size, wrapper_class=padding)
    return dataset


def init_dataset(data_type, data_list_file, conf=None, partition=True):
    return Dataset(data_type, data_list_file, conf, partition)
