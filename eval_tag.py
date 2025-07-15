"""eval mm model using a set of correct and incorrect tags as input hints"""
from functools import reduce
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import torch

from as_data import read_tags, read_events
from common import read_yaml, init_log
from dataset import RandomTagDataset, collate_multimodal_test
from eval_multimodal import test_epoch
from models.utils import model_size
from utils import read_labels
from utils import load_tokenizer
from utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')
    parser.add_argument('--ckpt', help='model checkpoint', default=None)
    parser.add_argument('--final', help='use final model', action='store_true')
    parser.add_argument('-o', '--output', default='results', help='output directory (default: results)')
    parser.add_argument('-n', dest='n_incorrect', type=int, default=0, help='number of incorrect tags (default: 0)')
    parser.add_argument('-w', '--weak', action='store_true', help='use audioset weak tags (default: use audioset strong as weak)')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='number of repeats')

    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    logger = init_log('eval-tags')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))

    hop_length = cfg.get('hop_length')
    sample_rate = cfg.get('sample_rate')
    hop_length_seconds = hop_length / sample_rate

    as_meta_dir = Path(cfg.get('audioset_meta_dir'))
    cache_dir = Path(cfg.get('cache_dir', 'cache'))

    batch_size = cfg.get('batch_size', 32)
    num_workers = cfg.get('num_dataloader_workers')

    if args.ckpt is not None:
        model_path = args.ckpt
    else:
        checkpoint_dir = Path(cfg.get('checkpoint_dir', 'ckpt'))
        if args.final:
            model_path = checkpoint_dir / 'ckpt-final.pt'
        else:
            model_path = checkpoint_dir / 'ckpt-best.pt'

    logger.info(f'loading model from {model_path}')
    model = load_checkpoint(model_path)
    logger.info(f'model size {model_size(model) / 1e6:.1f}M')
    model.eval()
    model = model.to(device)

    # get downsampling factor from the model config
    ds_factor = reduce(lambda x, y: x * y[1], model.config.poolings, 1)

    if args.weak:
        tags, _ = read_tags(as_meta_dir / 'audioset_weak_train_unbalanced.tsv', drop_time=True)
        tags2, _ = read_tags(as_meta_dir / 'audioset_weak_train_unbalanced2.tsv', drop_time=True)
        tags.update(tags2)
    else:
        tags, _ = read_tags(as_meta_dir / 'audioset_strong_train.tsv', drop_time=True)

    events, _ = read_events(as_meta_dir / 'audioset_strong_train.tsv', drop_time=True)
    labels, label_names = read_labels(cache_dir / 'labels.csv')

    # load tokenizer for the dataset: the tag name lists are tokenized on the fly
    tokenizer = load_tokenizer(cfg.get('tokenizer_dir', 'tokenizer'))

    test_idx = np.load(cache_dir / 'test_idx.npy')

    ds = RandomTagDataset(
        data_filename=cache_dir / 'train_data.hdf',
        file_idx=test_idx,
        tags=tags,
        labels=labels,
        class_names=label_names,
        tokenizer=tokenizer,
        n_incorrect=args.n_incorrect,
        downsampling_factor=ds_factor)

    test_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_multimodal_test)
    x, y, ym, z, fn = next(iter(test_loader))

    #label_ind = {label: ind for ind, label in enumerate(labels)}

    label_data = {mid: {'label': label} for mid, label in label_names.items()}
    overall = []
    for r in range(args.repeat):
        logger.info(f'begin round {r + 1}/{args.repeat}')
        #psds, classwise, sbased, aps, aucs = test_epoch(model, test_loader, events, labels, hop_length_seconds, device, ignore_captions=False)
        #logger.info(f'round {r + 1} - PSDS={psds:.4f}, mAP={aps.mean().item():.4f}, AUC={aucs.mean().item():.4f}')
        #print(sbased.keys())
    
        #psds, classwise, cw_sbased, overall_sbased, aps, aucs = test_epoch(model, test_loader, events, labels, hop_length_seconds, device, ignore_captions=False)
        psds1, _, classwise, _, cw_sbased, overall_sbased, aps, aucs = test_epoch(model, test_loader, events, labels, hop_length_seconds, device=device, ignore_captions=False, ignore_psds2=True)
        fnr = 1 - overall_sbased['sensitivity']
        fpr = 1 - overall_sbased['specificity']
        logger.info(f'PSDS-1: {psds1:.5f} mAP: {aps.mean().item():.5f} AUC: {aucs.mean().item():.5f} FNR: {fnr:.5f} FPR: {fpr:.5f}')
        fsc = overall_sbased['f_measure']
        pre = overall_sbased['precision']
        rec = overall_sbased['recall']
        logger.info(f'segment based F-score: {fsc:.5f} precision: {pre:.5f} recall: {rec:.5f}')
        overall.append([psds1, aps.mean().item(), aucs.mean().item(), fnr, fpr, fsc, pre, rec])

        for mid, cw_psds in classwise:
            label = label_names.get(mid)
            #rows.append((mid, label, cw_psds, aps[label_ind[mid]].item(), aucs[label_ind[mid]].item()))
            label_data[mid][f'psds_r{r + 1}'] = cw_psds
            label_data[mid][f'fsc_r{r + 1}'] = cw_sbased['f_measure'][mid]
            label_data[mid][f'pre_r{r + 1}'] = cw_sbased['precision'][mid]
            label_data[mid][f'rec_r{r + 1}'] = cw_sbased['recall'][mid]

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(overall, columns=['PSDS1', 'mAP', 'AUC', 'FNR', 'FPR', 'f_measure', 'precision', 'recall'])
    df.to_csv(output_dir / 'overall.csv')

    df = pd.DataFrame.from_dict(label_data, orient='index')
    df.index.name = 'mid'
    df.to_csv(output_dir / 'tag_results.csv')

if __name__ == '__main__':
    main()
