"""
create train/val/test splits of the data
"""
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from ac_data import read_captions
from common import read_yaml, read_json
from utils import data_fnames, read_tokens


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)

    ac_dir = Path(cfg.get('audiocaps_dir'))
    cache_dir = Path(cfg.get('cache_dir', 'cache'))

    validation_size = cfg.get('validation_size', 0.1)

    print('reading tokens')
    tokens = read_json(cache_dir / 'soundve-caps.json')

    print('reading data filenames')
    fnames = data_fnames(cache_dir / 'train_data.hdf')

    print('reading audiocaps data')
    captions = read_captions(ac_dir / 'dataset' / 'train.csv')
    ac_fns = set(captions.keys())

    # take only those files that we have tokens for
    tokenized_fnames = set(tokens.keys()).intersection(fnames)

    # for testing cut out those files that are in audiocaps train
    train_fns = tokenized_fnames - ac_fns
    train_fns = sorted(train_fns)
    train_fns, val_fns = train_test_split(train_fns, test_size=validation_size)
    test_fns = tokenized_fnames.intersection(ac_fns)
    train_fns = set(train_fns)
    val_fns = set(val_fns)
    print(f'training: {len(train_fns)}, validation: {len(val_fns)}, testing: {len(test_fns)}')

    # write names
    (cache_dir / 'train_fns.txt').write_text('\n'.join(sorted(train_fns)))
    (cache_dir / 'val_fns.txt').write_text('\n'.join(sorted(val_fns)))
    (cache_dir / 'test_fns.txt').write_text('\n'.join(sorted(test_fns)))

    # write indices (for faster access)
    train_idx = np.array([k for k, fn in enumerate(fnames) if fn in train_fns], dtype=np.int32)
    val_idx = np.array([k for k, fn in enumerate(fnames) if fn in val_fns], dtype=np.int32)
    test_idx = np.array([k for k, fn in enumerate(fnames) if fn in test_fns], dtype=np.int32)
    np.save(cache_dir / 'train_idx.npy', train_idx)
    np.save(cache_dir / 'val_idx.npy', val_idx)
    np.save(cache_dir / 'test_idx.npy', test_idx)


if __name__ == '__main__':
    main()
