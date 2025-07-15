
import argparse
import numpy as np
import pandas as pd
import time
import torch
import torchmetrics

from functools import reduce
from pathlib import Path
from sed_scores_eval import intersection_based, segment_based
from sed_scores_eval.base_modules.scores import create_score_dataframe
import sed_eval
import dcase_util

from as_data import read_events
from common import read_yaml, init_log
from dataset import SEDCaptionTestDataset, collate_multimodal_test
from models.utils import model_size
from utils import read_labels, read_tokens, data_fnames
from utils import make_event_list
from utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')
    parser.add_argument('--logdir', help='tensorboard logdir')
    parser.add_argument('--ckpt', help='model checkpoint', default=None)
    parser.add_argument('--final', help='use final model', action='store_true')
    parser.add_argument('-o', '--output', default='results', help='output directory (default: results)')
    #parser.add_argument('-d', '--downsampling', default=1, type=int, help='downsampling factor')
    parser.add_argument('-i', dest='ignore_captions', help='ignore captions', action='store_true', default=False)
    parser.add_argument('-s', '--soundve', action='store_true', help='use SoundVE captions', default=False)
    parser.add_argument('-t', dest='tokens', type=str, help='token json filename for captions', default=None)

    return parser.parse_args()



@torch.inference_mode()
def test_epoch(model, loader, events, labels, hop_length_seconds, device, eval_labels: list[str] | None = None, ignore_captions=False, ignore_psds2=False):
    model.eval()
    eval_labels = eval_labels if eval_labels is not None else labels
    n_labels = len(eval_labels)
    downsampling_factor = loader.dataset.downsampling_factor

    if n_labels > 1:
        ap_metric = torchmetrics.AveragePrecision(task='multilabel', num_labels=n_labels, average='none')
        auc_metric = torchmetrics.AUROC(task='multilabel', num_labels=n_labels, average='none')
    else:
        ap_metric = torchmetrics.AveragePrecision(task='binary', average='none')
        auc_metric = torchmetrics.AUROC(task='binary', average='none')

    # psds-1
    dtc_threshold = 0.7
    gtc_threshold = 0.7
    cttc_threshold = None
    alpha_st = 0.0 # 1.0 in orig
    alpha_ct = 0.0

    # psds-2
    dtc_th2 = 0.1
    gtc_th2 = 0.1
    cttc_th2 = 0.3
    alpha_st2 = 0.0 # 1.0 in orig
    alpha_ct2 = 0.5

    pred = {}
    ref = {}
    durations = {}

    # time resolution for the segment based
    #tr = 0.3
    #tr = 24 * hop_length_seconds * downsampling_factor
    tr = 1.0
    #print(f'time resolution = {tr:.2f} s')

    label_ind = torch.as_tensor([labels.index(label) for label in eval_labels], dtype=torch.int64)

    # use original sed_eval for fdr/fnr
    sed_sbm = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=labels, time_resolution=tr)

    # count framewise fp/fn for each class
    fp = torch.zeros(n_labels, n_labels)
    fn = torch.zeros(n_labels, n_labels)
    n_pos = torch.zeros(n_labels)
    n_neg = torch.zeros(n_labels)

    for x, y, y_mask, z_true, fnames in loader:
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)

        if ignore_captions:
            y = None
            y_mask = None

        z_pred = model(audio=x, text=y, text_mask=y_mask).cpu()
        z_pred = z_pred[:, :, label_ind]
        z_pred = z_pred.sigmoid()

        z_true = z_true[:, :, label_ind]

        # reshape for torchmetrics
        zp = z_pred.reshape(-1, z_pred.size(-1))
        zt = z_true.reshape(-1, z_true.size(-1)).int()

        ap_metric.update(zp, zt)
        auc_metric.update(zp, zt)

        # various sed stuff
        timestamps = downsampling_factor * hop_length_seconds * np.arange(z_pred.size(1) + 1)

        for k in range(x.size(0)):
            fn = fnames[k]

            ref_events = [event for event in events[fn] if event[2] in eval_labels]
            #if len(ref_events) == 0:
            #    continue

            ref[fn] = ref_events
            pred[fn] = create_score_dataframe(
                scores=z_pred[k].numpy(),
                timestamps=timestamps,
                event_classes=eval_labels)
            durations[fn] = 10.0

            # construct data for sed_eval
            zk = z_pred[k]
            refs = []
            preds = []
            for onset, offset, label in make_event_list(zk, labels, hop_length_seconds, threshold=0.5):
                event = {
                    'event_label': label,
                    'event_onset': onset,
                    'event_offset': offset,
                    'file': fn,
                }

                preds.append(event)

            for onset, offset, label in events[fn]:
                if label not in labels:
                    continue

                event = {
                    'event_label': label,
                    'event_onset': onset,
                    'event_offset': offset,
                    'file': fn,
                }

                refs.append(event)

            refs = dcase_util.containers.MetaDataContainer(refs)
            preds = dcase_util.containers.MetaDataContainer(preds)

            sed_sbm.evaluate(reference_event_list=refs, estimated_event_list=preds)

    t0 = time.perf_counter()
    psds1, single_class_psds, psd_roc, single_class_psd_rocs = intersection_based.psds(
        scores=pred,
        ground_truth=ref,
        audio_durations=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        alpha_ct=alpha_ct,
        alpha_st=alpha_st,
        unit_of_time='hour',
        max_efpr=100.0,
        num_jobs=24)
    dt = time.perf_counter() - t0
    print(f'psds1: {dt:.1f} s')
    print('psds:', psds1)
    print('single class avg:', np.mean(list(single_class_psds.values())))

    classwise = list(single_class_psds.items())
    classwise.sort()

    if not ignore_psds2:
        t0 = time.perf_counter()
        psds2, single_class_psds2, _, _ = intersection_based.psds(
            scores=pred,
            ground_truth=ref,
            audio_durations=durations,
            dtc_threshold=dtc_th2,
            gtc_threshold=gtc_th2,
            cttc_threshold=cttc_th2,
            alpha_ct=alpha_ct2,
            alpha_st=alpha_st2,
            unit_of_time='hour',
            max_efpr=100.0,
            num_jobs=24)
        dt = time.perf_counter() - t0
        print(f'psds2: {dt:.1f} s')
        print('psds:', psds2)
        print('single class avg:', np.mean(list(single_class_psds2.values())))
        classwise_psds2 = single_class_psds2
    else:
        psds2 = None
        classwise_psds2 = None

    thresholds = {label: 0.5 for label in labels}
    fsc, pre, rec, stats = segment_based.fscore(
        scores=pred,
        ground_truth=ref,
        audio_durations=durations,
        threshold=thresholds,
        segment_length=1.0,
        num_jobs=20)

    aps = ap_metric.compute()
    aucs = auc_metric.compute()

    sb_results = sed_sbm.results()
    fsc = {label: sb_results['class_wise'][label]['f_measure']['f_measure'] for label in labels}
    pre = {label: sb_results['class_wise'][label]['f_measure']['precision'] for label in labels}
    rec = {label: sb_results['class_wise'][label]['f_measure']['recall'] for label in labels}
    sens = {label: sb_results['class_wise'][label]['accuracy']['sensitivity'] for label in labels}
    spec = {label: sb_results['class_wise'][label]['accuracy']['specificity'] for label in labels}

    cw_sbased = {
        'f_measure': fsc,
        'precision': pre,
        'recall': rec,
        'sensitivity': sens,
        'specificity': spec,
    }

    overall_sbased = {
        'f_measure': sb_results['overall']['f_measure']['f_measure'],
        'precision': sb_results['overall']['f_measure']['precision'],
        'recall': sb_results['overall']['f_measure']['recall'],
        'sensitivity': sb_results['overall']['accuracy']['sensitivity'],
        'specificity': sb_results['overall']['accuracy']['specificity'],
    }

    return psds1, psds2, classwise, classwise_psds2, cw_sbased, overall_sbased, aps, aucs


@torch.inference_mode()
def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    logger = init_log('eval-crnn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    data_fn = cache_dir / 'train_data.hdf'

    hop_length = cfg.get('hop_length')
    sample_rate = cfg.get('sample_rate')
    hop_length_seconds = hop_length / sample_rate

    if args.ignore_captions:
        logger.info('ignoring captions for evaluation')

    batch_size = cfg.get('batch_size', 16)
    num_workers = cfg.get('num_dataloader_workers', 8)

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
    model = model.to(device)
    model.eval()

    labels, label_names = read_labels(cache_dir / 'labels.csv')
    events, _ = read_events(Path(cfg.get('audioset_meta_dir')) / 'audioset_strong_train.tsv', drop_time=True)
    label_ind = {label: k for k, label in enumerate(labels)}

    # get eval fnames
    fnames = data_fnames(data_fn)
    test_idx = np.load(cache_dir / 'test_idx.npy')
    eval_fns = [fnames[k] for k in test_idx]

    if args.tokens is not None:
        # load custom tokens
        logger.info(f'loading tokens from {args.tokens}')
        eval_tokens = read_tokens(args.tokens)
    elif args.soundve:
        logger.info('using SoundVE captions')
        eval_tokens = read_tokens(cache_dir / 'soundve-caps.json')
    else:
        # the default is to use audiocaps tokens for eval
        eval_token_fn = cfg.get('eval_token_fn', cache_dir / 'audiocaps.json')
        eval_tokens = read_tokens(eval_token_fn)

    # do the filtering by the filenames in the split
    eval_tokens = {fn: tokens for fn, tokens in eval_tokens.items() if fn in eval_fns}
    logger.info(f'got tokens for {len(eval_tokens)} files')

    logger.info(f'data has {len(test_idx)} files for evaluation')

    # downsampling factor
    ds_factor = reduce(lambda x, y: x * y[1], model.config.poolings, 1)
    ds = SEDCaptionTestDataset(data_fn, file_idx=test_idx, tokens=eval_tokens, downsampling_factor=ds_factor)
    test_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_multimodal_test)

    psds1, psds2, classwise, classwise2, cw_sbased, overall_sbased, aps, aucs = test_epoch(model, test_loader, events, labels, hop_length_seconds, device, ignore_captions=args.ignore_captions, ignore_psds2=False)
    logger.info(f'PSDS-1: {psds1:.5f} PSDS-2: {psds2:.5f} AUC: {aucs.mean().item():.5f} mAP: {aps.mean().item():.5f}')
    #logger.info(f'PSDS-1: {psds1:.5f} AUC: {aucs.mean().item():.5f} mAP: {aps.mean().item():.5f} FNR: {fnr:.5f} FPR: {fpr:.5f}')

    fnr = 1 - overall_sbased['sensitivity']
    fpr = 1 - overall_sbased['specificity']
    fsc = overall_sbased['f_measure']
    pre = overall_sbased['precision']
    rec = overall_sbased['recall']
    logger.info(f'segment based F-score: {fsc:.5f} precision: {pre:.5f} recall: {rec:.5f} FNR: {fnr:.5f} FPR: {fpr:.5}')
    #print(ib_results)

    rows = []
    for mid, cw_psds in classwise:
        cw_psds2 = classwise2[mid]
        label = label_names.get(mid)
        fsc = cw_sbased['f_measure'][mid]
        pre = cw_sbased['precision'][mid]
        rec = cw_sbased['recall'][mid]
        fnr = 1 - cw_sbased['sensitivity'][mid]
        fpr = 1 - cw_sbased['specificity'][mid]
        rows.append((mid, label, cw_psds, cw_psds2, aps[label_ind[mid]].item(), aucs[label_ind[mid]].item(), fsc, pre, rec, fnr, fpr))

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    if args.ignore_captions:
        out_fn = 'nocaps.csv'
    elif args.tokens is not None:
        out_fn = Path(args.tokens).stem + '.csv'
    elif args.soundve:
        out_fn = 'soundve.csv'
    else:
        out_fn = 'ac.csv'
    df = pd.DataFrame(rows, columns=['mid', 'label', 'psds1', 'psds2', 'ap', 'auc', 'f-score', 'precision', 'recall', 'fnr', 'fpr']).set_index('mid')
    df.to_csv(output_dir / out_fn)


if __name__ == '__main__':
    main()


