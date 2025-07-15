from pathlib import Path
from tqdm import tqdm

import argparse
import gc
import numpy as np
import time
import torch
import torch.nn.functional as F
import torchmetrics

from sed_scores_eval import intersection_based
from sed_scores_eval.base_modules.scores import create_score_dataframe

from sklearn.model_selection import train_test_split

from as_data import read_events
from common import init_log, read_yaml
#from dataset import SEDCaptionDataset, collate_multimodal
from dataset import SEDCaptionTagDataset, collate_multimodal_tag
from dataset import SEDCaptionTestDataset, collate_multimodal_test
from models.tf import MultimodalCRNNConfig, MultimodalCRNN
from models.utils import model_size
from sampling import get_sample_probabilities
from train_utils import batch_pos_weight, spectrogram_mixing, entropy
from train_utils import optimizer_size
from utils import read_labels, read_tokens


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')
    parser.add_argument('--logdir', help='tensorboard logdir')
    parser.add_argument('--ckpt', help='continue from checkpoint')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='number of repeats')

    return parser.parse_args()


def collect_tag_probabilities(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    tags = []

    for _, _, _, z_true, _ in tqdm(loader):
        z_tags, _ = torch.max(z_true, dim=1)
        tags.append(z_tags)

    tags = torch.cat(tags, dim=0)
    return torch.mean(tags, dim=0)


def train_epoch(model: torch.nn.Module,
                loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.OneCycleLR | None = None,
                clip_grad_norm: float | None = None,
                grad_acc_steps: int = 1,
                p_mask_tokens: float = 0.0,
                p_dropout_caption: float = 0.0,
                mask_token_id: int = 4,
                batch_loss_weighting_strong: bool = False,
                batch_loss_weighting_weak: bool = False,
                pos_weight_strong: float | torch.Tensor | None = None,
                pos_weight_weak: float | torch.Tensor | None = None,
                spectrogram_mixing_param: float | None = None,
                ignore_captions: bool = False,
                log_interval: int | None = None,
                device: torch.device | None = None) -> float:
    model.train()
    batch_t0 = time.perf_counter()
    train_loss = 0.0

    # batch loss weighting overrides

    if isinstance(pos_weight_strong, float):
        pos_weight_strong = torch.as_tensor(pos_weight_strong, device=device)
    elif isinstance(pos_weight_strong, torch.Tensor):
        pos_weight_strong = pos_weight_strong.to(device)

    if isinstance(pos_weight_weak, float):
        pos_weight_weak = torch.as_tensor(pos_weight_weak, device=device)
    elif isinstance(pos_weight_strong, torch.Tensor):
        pos_weight_weak = pos_weight_weak.to(device)

    device_type = device.type if device is not None else 'cpu'
    scaler = torch.amp.GradScaler(device)

    optimizer.zero_grad()
    for batch, (x, y, y_mask, z_true, weak_idx) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        z_true = z_true.to(device)
        weak_idx = weak_idx.to(device)

        if ignore_captions:
            y = None
            y_mask = None
        elif p_mask_tokens > 0:
            idx = torch.rand(y.shape, device=y.device) < p_mask_tokens
            y[idx & (y > mask_token_id)] = mask_token_id

            if p_dropout_caption > 0:
                idx = torch.rand(x.size(0)) < p_dropout_caption
                # which one?
                y[idx, :] = mask_token_id
                y_mask[idx, :] = True

        use_weak = torch.any(weak_idx)
        if use_weak:
            z_true_weak, _ = torch.max(z_true, dim=1)
        else:
            z_true_weak = None

        if spectrogram_mixing_param is not None and spectrogram_mixing_param > 0:
            x = spectrogram_mixing(x, max_amount=spectrogram_mixing_param)

        if batch_loss_weighting_strong or batch_loss_weighting_weak:
            w = batch_pos_weight(z_true)

            if batch_loss_weighting_strong:
                pos_weight_strong = w

            if batch_loss_weighting_weak:
                pos_weight_weak = w

        with torch.amp.autocast(device_type):
            if use_weak:
                z_pred_strong, z_pred_weak = model(audio=x, text=y, text_mask=y_mask, return_weak=True)
                weak_loss = F.binary_cross_entropy_with_logits(z_pred_weak[weak_idx], z_true_weak[weak_idx], pos_weight=pos_weight_weak)
                #weak_loss = F.binary_cross_entropy_with_logits(z_pred_weak, z_true_weak)
                strong_loss = F.binary_cross_entropy_with_logits(z_pred_strong[~weak_idx], z_true[~weak_idx], pos_weight=pos_weight_strong)
            else:
                z_pred_strong = model(audio=x, text=y, text_mask=y_mask)
                weak_loss = torch.tensor(0.0, device=x.device)
                #weak_loss = F.binary_cross_entropy_with_logits(z_pred_weak, z_true_weak)
                strong_loss = F.binary_cross_entropy_with_logits(z_pred_strong, z_true, pos_weight=pos_weight_strong)

            loss = strong_loss + weak_loss

        train_loss += loss.item()
        scaler.scale(loss / grad_acc_steps).backward()

        if (batch + 1) % grad_acc_steps == 0:
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if log_interval is not None and batch % log_interval == 0:
            t_batch = int(1000 * (time.perf_counter() - batch_t0) / log_interval)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'batch {batch:4d}/{len(loader)} - {t_batch} ms/batch - lr {current_lr:.4g} - training loss {loss.item():.4f} (strong {strong_loss.item():.4f}, weak {weak_loss.item():.4f})')
            batch_t0 = time.perf_counter()

    return train_loss / len(loader)


@torch.inference_mode()
def test_epoch_loss(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    ignore_captions: bool = False,
                    device: torch.device | None = None) -> tuple[float, float]:

    model.eval()
    test_loss = 0.0

    n_labels = model.config.n_classes
    ap_metric = torchmetrics.AveragePrecision(task='multilabel', num_labels=n_labels)

    for x, y, y_mask, z_true, _ in loader:
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        z_true = z_true.to(device)

        if ignore_captions:
            y = None
            y_mask = None

        z_pred = model(audio=x, text=y, text_mask=y_mask)
        test_loss += F.binary_cross_entropy_with_logits(z_pred, z_true).item()

        zp = z_pred.reshape(-1, z_pred.size(-1))
        zt = z_true.reshape(-1, z_true.size(-1)).int()

        ap_metric.update(zp, zt)

    return test_loss / len(loader), ap_metric.compute().item()


@torch.inference_mode()
def test_epoch_psds(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    labels: list[str],
                    events: dict[str, list[tuple[float, float, str]]],
                    hop_length_seconds: float,
                    label_names: dict[str, str] | None = None,
                    n_best: int = 10,
                    ignore_captions: bool = False,
                    device: torch.device | None = None) -> tuple[float, float]:
    # note: this is currently not used
    model.eval()
    test_loss = 0.0

    # psds-1
    dtc_threshold = 0.7
    gtc_threshold = 0.7
    cttc_threshold = None
    alpha_st = 1.0
    alpha_ct = 0.0

    pred = {}
    ref = {}
    durations = {}

    ds_factor = model.config.audio_enc_stack_size

    for x, y, y_mask, z_true, fnames in loader:
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        z_true = z_true.to(device)

        if ignore_captions:
            y = None
            y_mask = None

        z_pred = model(audio=x, text=y, text_mask=y_mask)
        test_loss += F.binary_cross_entropy_with_logits(z_pred, z_true).item()

        z_pred = z_pred.sigmoid().cpu()
        timestamps = ds_factor * hop_length_seconds * np.arange(z_pred.size(1) + 1)

        for k, fn in enumerate(fnames):
            ref_events = [event for event in events[fn] if event[2] in labels]
            if len(ref_events) == 0:
                continue

            ref[fn] = ref_events
            pred[fn] = create_score_dataframe(
                scores=z_pred[k].numpy(),
                timestamps=timestamps,
                event_classes=labels)

            durations[fn] = timestamps[-1]

    psds, single_class_psds, psd_roc, single_class_psd_rocs = intersection_based.psds(
        scores=pred,
        ground_truth=ref,
        audio_durations=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        alpha_ct=alpha_ct,
        alpha_st=alpha_st,
        unit_of_time='hour',
        max_efpr=100.0)

    if label_names is not None:
        # print out the best ones
        classwise = sorted(single_class_psds.items(), key=lambda x: x[1], reverse=True)
        for mid, cw_psds in classwise[:n_best]:
            label = label_names.get(mid, mid)
            print(f'{label:40s} {cw_psds:.4f}')

    return test_loss / len(loader), psds


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    logger = init_log('train-multimodal')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    data_fn = cache_dir / 'train_data.hdf'

    mcfg = cfg.get('crnn_multimodal', {})
    batch_size = mcfg.get('batch_size', cfg.get('batch_size', 16))
    num_workers = cfg.get('num_dataloader_workers', 8)

    data_resampling = mcfg.get('data_resampling', cfg.get('data_resampling', False))
    global_loss_weighting = mcfg.get('global_loss_weighting', cfg.get('global_loss_weighting', False))
    batch_loss_weighting = mcfg.get('batch_loss_weighting', cfg.get('batch_loss_weighting', False))
    spectrogram_mixing_param = mcfg.get('spectrogram_mixing_param')

    learning_rate = mcfg.get('learning_rate', cfg.get('learning_rate', 1e-4))
    weight_decay = mcfg.get('weight_decay', cfg.get('weight_decay', 1e-2))
    clip_grad_norm = mcfg.get('clip_grad_norm', cfg.get('clip_grad_norm'))
    grad_acc_steps = mcfg.get('grad_acc_steps', cfg.get('grad_acc_steps', 1))
    warmup_pct = mcfg.get('warmup_pct', cfg.get('warmup_pct', 0.0))

    n_epochs = mcfg.get('n_epochs', cfg.get('n_epochs', 50))
    log_interval = mcfg.get('log_interval', cfg.get('log_interval'))

    checkpoint_dir = Path(mcfg.get('checkpoint_dir', cfg.get('checkpoint_dir', 'ckpt')))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    events, _ = read_events(Path(cfg.get('audioset_meta_dir')) / 'audioset_strong_train.tsv', drop_time=True)
    labels, label_names = read_labels(cache_dir / 'labels.csv')
    n_classes = len(labels)

    # for psds validation (skip for now)
    hop_length_seconds = cfg.get('hop_length') / cfg.get('sample_rate')
    eval_psds_after_epoch = mcfg.get('eval_psds_after_epoch')

    # amount of weakly labeled data
    weak_proportion = mcfg.get('weak_proportion', 0.0)

    # we need this to downsample the targets
    audio_enc_stack_size = mcfg.get('audio_enc_stack_size', 4)

    caption_dropout = mcfg.get('caption_dropout', 0.0)
    logger.info(f'{caption_dropout=}')
    ignore_captions = mcfg.get('ignore_captions', False)
    ignore_caption_epochs = mcfg.get('ignore_caption_epochs', 0)
    if ignore_captions:
        logger.info('ignoring captions for training')
    elif ignore_caption_epochs > 0:
        logger.info(f'ignoring captions for the first {ignore_caption_epochs} epochs')

    logger.info('reading tokens')
    train_token_fn = cfg.get('train_token_fn', cache_dir / 'soundve-caps.json')
    train_tokens = read_tokens(train_token_fn)
    logger.info(f'got tokens for {len(train_tokens)} files')

    train_idx = np.load(cache_dir / 'train_idx.npy')
    val_idx = np.load(cache_dir / 'val_idx.npy')

    gt = None

    if data_resampling:
        logger.info('using resampling, calculating sample probabilities')
        gt = np.load(cache_dir / 'train_gt.npy')
        gt = torch.as_tensor(gt)
        probs = get_sample_probabilities(gt, offset=0.01, eps=1e-8)
        sampler = torch.utils.data.WeightedRandomSampler(probs, num_samples=len(train_idx), replacement=True)

        # expected output:
        h_orig = entropy(gt.mean(dim=0))
        expected = probs @ gt
        expected = expected / expected.sum()
        h_exp = entropy(expected)
        logger.info(f'expected tag entropy {h_exp:.4f} (original gt {h_orig:.4f}, max {np.log2(n_classes):.4f})')
    else:
        sampler = None

    pos_weight = None
    if global_loss_weighting:
        if gt is None:
            gt = np.load(cache_dir / 'train_gt.npy')
            gt = torch.as_tensor(gt)

        p_pos = gt.mean(dim=0)
        pos_weight = (1 - p_pos) / p_pos

    if weak_proportion > 0:
        _, weak_idx = train_test_split(train_idx, test_size=weak_proportion, random_state=303)
    else:
        weak_idx = []
    logger.info(f'{len(train_idx)} files for training ({len(weak_idx)} weakly labeled), {len(val_idx)} for validation')

    #n_train_captioned = len(set(captioned_file_idx).intersection(train_idx))
    #n_val_captioned = len(set(captioned_file_idx).intersection(val_idx))
    #logger.info(f'{n_train_captioned}/{len(train_idx)} training files have captions, {n_val_captioned}/{len(val_idx)} validation files have captions')

    ds_train = SEDCaptionTagDataset(
        data_filename=data_fn,
        file_idx=train_idx,
        weak_idx=weak_idx,
        downsampling_factor=audio_enc_stack_size,
        tokens=train_tokens,
        slice_swapping=True)

    ds_val = SEDCaptionTestDataset(
        data_filename=data_fn,
        file_idx=val_idx,
        downsampling_factor=audio_enc_stack_size,
        tokens=train_tokens)

    train_loader = torch.utils.data.DataLoader(
        dataset=ds_train,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_multimodal_tag)

    val_loader = torch.utils.data.DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_multimodal_test)

    p_text_enc_masking = mcfg.get('p_text_enc_masking', 0.0)

    # model
    model_cfg = MultimodalCRNNConfig(
        n_classes=n_classes,

        dims=mcfg.get('dims', [16, 32, 64, 128, 128, 128]),
        kernel_sizes=mcfg.get('kernel_sizes'),
        poolings=mcfg.get('poolings'),
        cnn_dropout=mcfg.get('cnn_dropout', 0.1),
        n_time_mask=mcfg.get('n_time_mask', 0),
        n_freq_mask=mcfg.get('n_freq_mask', 0),
        time_mask_param=mcfg.get('time_mask_param', 25),
        freq_mask_param=mcfg.get('freq_mask_param', 8),

        # mean, panns, attn
        weak_pooling=mcfg.get('weak_pooling', 'panns'),

        # tf
        vocab_size=cfg.get('vocab_size'),
        d_tf=mcfg.get('d_tf', 128),
        n_tf_layers=mcfg.get('n_text_enc_layers', 4),
        n_tf_heads=mcfg.get('n_text_enc_heads', 8),
        p_text_masking=mcfg.get('p_text_masking', 0.0),
        tf_dropout=mcfg.get('tf_dropout', 0.0),

        ca_dropout=mcfg.get('ca_dropout', 0.0),

        # rnn
        n_rnn_layers=mcfg.get('n_rnn_layers', 2),
        rnn_dropout=mcfg.get('rnn_dropout', 0.0),
        d_rnn=mcfg.get('d_rnn', 64))

    print(model_cfg)

    for r in range(args.repeat):
        if args.repeat > 1:
            logger.info(f'begin round {r + 1}/{args.repeat}')

        model = MultimodalCRNN(model_cfg)
        logger.info(f'model size {model_size(model) / 1e6:.1f}M')
        logger.info(f'CNN size {model_size(model.cnn) / 1e6:.1f}M, text encoder size {model_size(model.text_encoder) / 1e6:.1f}M')

        if args.ckpt is not None:
            logger.info(f'loading model checkpoint from {args.ckpt}')
            ckpt = torch.load(args.ckpt, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])

        model.train()
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=learning_rate,
            total_steps=n_epochs * len(train_loader),
            pct_start=warmup_pct)

        best_capt_map = 0.0
        best_nocapt_map = 0.0

        best_nocapt_epoch = None
        best_capt_epoch = None

        for epoch in range(n_epochs):
            epoch_ignore_captions = ignore_captions or epoch < ignore_caption_epochs

            # print(torch.cuda.memory_summary())
            # torch.cuda.reset_peak_memory_stats()

            train_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                spectrogram_mixing_param=spectrogram_mixing_param,
                batch_loss_weighting_strong=batch_loss_weighting,
                pos_weight_strong=pos_weight,
                pos_weight_weak=pos_weight,
                clip_grad_norm=clip_grad_norm,
                grad_acc_steps=grad_acc_steps,
                log_interval=log_interval,
                p_mask_tokens=p_text_enc_masking,
                p_dropout_caption=caption_dropout,
                mask_token_id=4,
                ignore_captions=epoch_ignore_captions,
                device=device)

            # always ignore captions for validation
            val_loss, val_map = test_epoch_loss(model=model, loader=val_loader, ignore_captions=True, device=device)

            # some mem info
            optimizer_state_size = optimizer_size(optimizer) / 1024 ** 2
            cuda_max_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
            logger.info(f'optimizer state {optimizer_state_size:.1f} MB, cuda max mem allocated {cuda_max_mem:.1f} MB')

            torch.cuda.empty_cache()
            gc.collect()

            if epoch_ignore_captions:
                logger.info(f'epoch {epoch + 1}/{n_epochs} - training loss {train_loss:.4f} - validation loss {val_loss:.4f} - frame mAP {val_map:.4f}')
            else:
                if eval_psds_after_epoch is not None:
                    val_loss_capt, val_psds = test_epoch_psds(model=model, loader=val_loader, ignore_captions=False, device=device)
                    logger.info(f'epoch {epoch + 1}/{n_epochs} - training loss {train_loss:.4f} - validation loss without/with captions: {val_loss:.4f} / {val_loss_capt:.4f} - psds (w/capt) {val_psds:.4f}')

                else:
                    val_loss_capt, val_map_capt = test_epoch_loss(model=model, loader=val_loader, ignore_captions=False, device=device)
                    logger.info(f'epoch {epoch + 1}/{n_epochs} - training loss {train_loss:.4f} - validation loss without/with captions: {val_loss:.4f} / {val_loss_capt:.4f} - frame mAP {val_map:.4f} / {val_map_capt:.4f}')

                #if val_loss_capt < best_capt_loss:
                if val_map_capt > best_capt_map:
                    #best_capt_loss = val_loss_capt
                    best_capt_map = val_map_capt
                    best_capt_epoch = epoch + 1
                    ckpt = {
                        'config': model_cfg,
                        'state_dict': model.state_dict(),
                    }

                    torch.save(ckpt, checkpoint_dir / f'ckpt-best-capt-{r}.pt')

            if val_map > best_nocapt_map:
                best_nocapt_map = val_map
                best_nocapt_epoch = epoch + 1
                ckpt = {
                    'config': model_cfg,
                    'state_dict': model.state_dict(),
                }

                torch.save(ckpt, checkpoint_dir / f'ckpt-best-nocapt-{r}.pt')

        ckpt = {
            'config': model_cfg,
            'state_dict': model.state_dict(),
        }

        torch.save(ckpt, checkpoint_dir / f'ckpt-final-{r}.pt')
        
        if best_capt_epoch is not None:
            logger.info(f'best no-caption epoch {best_nocapt_epoch}, best epoch with captions {best_capt_epoch}')
        else:
            logger.info(f'best epoch {best_nocapt_epoch}')


if __name__ == '__main__':
    main()
