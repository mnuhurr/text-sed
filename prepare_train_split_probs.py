from pathlib import Path
from tqdm import tqdm

import argparse
import numpy as np

from as_data import read_tags
from common import init_log, read_yaml
from utils import data_fnames, data_sizes, read_labels


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')

    return parser.parse_args()


def collect_gt_matrix(filenames: list[str], tags: dict[str, list[str]], labels: list[str]) -> np.ndarray:
    label_idx = {label: k for k, label in enumerate(labels)}

    gt = np.zeros((len(filenames), len(labels)))

    for k, fn in enumerate(tqdm(filenames)):
        for tag in tags[fn]:
            if tag not in label_idx:
                continue

            j = label_idx[tag]
            gt[k, j] = 1

    return gt


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    logger = init_log('prepare-train-data')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))

    logger.info('reading tags')
    tags, _ = read_tags(Path(cfg.get('audioset_meta_dir')) / 'audioset_strong_train.tsv', drop_time=True)
    labels, label_names = read_labels(cache_dir / 'labels.csv')

    train_data_fn = cache_dir / 'train_data.hdf'
    n_files, n_mels, n_classes = data_sizes(train_data_fn)
    train_data_fnames = data_fnames(train_data_fn)

    # take thos files that have a caption
    train_idx = np.load(cache_dir / 'train_idx.npy')
    val_idx = np.load(cache_dir / 'val_idx.npy')
    test_idx = np.load(cache_dir / 'test_idx.npy')
    assert len(set(test_idx).intersection(train_idx)) == 0
    assert len(set(test_idx).intersection(val_idx)) == 0
    logger.info(f'{len(train_idx)} files for training, {len(val_idx)} for validation, {len(test_idx)} for testing')

    train_fns = [train_data_fnames[k] for k in train_idx]
    val_fns = [train_data_fnames[k] for k in val_idx]

    train_gt = collect_gt_matrix(train_fns, tags, labels)
    val_gt = collect_gt_matrix(val_fns, tags, labels)

    np.save(cache_dir / 'train_gt.npy', train_gt)
    np.save(cache_dir / 'val_gt.npy', val_gt)

    (cache_dir / 'fnames_train.txt').write_text('\n'.join(train_fns))
    (cache_dir / 'fnames_val.txt').write_text('\n'.join(val_fns))


if __name__ == '__main__':
    main()
