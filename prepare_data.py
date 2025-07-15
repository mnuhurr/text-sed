import csv
import h5py
import numpy as np
import librosa

from pathlib import Path
from tqdm import tqdm

from as_data import read_labels, read_tags, read_events, read_ac_captions, collect_audio_filenames
from common import read_yaml, init_log


def povey_window(window_length: int) -> np.ndarray:
    w = 2 * np.pi * np.arange(window_length) / (window_length - 1)
    return (0.5 - 0.5 * np.cos(w))**0.85


def top_tags(tags: dict[str, list[str]], n_labels: int) -> dict[str, int]:
    count = {}
    for taglist in tags.values():
        for tag in taglist:
            if tag not in count:
                count[tag] = 1
            else:
                count[tag] += 1

    label_list = [(c, label) for label, c in count.items()]
    label_list.sort(reverse=True)

    return label_list[:n_labels]


def write_label_info(filename: str | Path, label_count_list: list[tuple[int, str]], label_names: dict[str, str]):
    with Path(filename).open('wt') as f:
        writer = csv.writer(f)

        writer.writerow(['mid', 'label', 'count'])

        for count, mid in label_count_list:
            label = label_names.get(mid, '')
            writer.writerow([mid, label, count])


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('prepare-data')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    cache_dir.mkdir(exist_ok=True, parents=True)

    ac_dir = Path(cfg.get('audiocaps_dir'))
    meta_dir = Path(cfg.get('audioset_meta_dir'))
    strong_tags, _ = read_tags(meta_dir / 'audioset_strong_train.tsv')

    label_names = read_labels(meta_dir / 'class_labels.tsv')
    n_labels = cfg.get('n_labels', 10)

    captions = read_ac_captions(ac_dir / 'dataset' / 'train.csv')

    common_keys = sorted(set(strong_tags.keys()).intersection(captions.keys()))
    logger.info(f'found {len(common_keys)} common filenames for audioset strong train and audiocaps train')

    sample_rate = cfg.get('sample_rate', 32000)
    n_fft = cfg.get('n_fft', 1024)
    hop_length = cfg.get('hop_length', 320)
    n_mels = cfg.get('n_mels', 64)
    f_min = cfg.get('f_min', 0.0)
    max_duration = cfg.get('max_duration', 10.0)
    min_duration = cfg.get('min_duration', 2.0)
    win_length = cfg.get('win_length')
    window = 'hann' if win_length is None else povey_window(win_length)

    # count tags
    common_tags = {fn: strong_tags[fn] for fn in common_keys}
    label_count = top_tags(common_tags, n_labels=n_labels)
    write_label_info(cache_dir / 'labels.csv', label_count, label_names)
    labels = sorted([label for c, label in label_count])
    label_ind = {label: k for k, label in enumerate(labels)}

    # collect audioset strong ids that contain events from any of the chosen classes
    strong_ids = [fn[:11] for fn, tags in strong_tags.items() if len(set(tags).intersection(labels)) > 0]

    events, _ = read_events(meta_dir / 'audioset_strong_train.tsv', drop_time=True)

    audio_filenames = collect_audio_filenames(Path(cfg.get('audioset_dir')) / 'audio', filter_set=strong_ids)
    n_files = len(audio_filenames)

    logger.info(f'{sample_rate=}, {n_fft=}, {hop_length=}, {n_mels=}, {f_min=}, {win_length=}')
    mel_len = int(sample_rate * max_duration) // hop_length + 1

    hop_len_seconds = hop_length / sample_rate

    with h5py.File(cache_dir / 'train_data.hdf', 'w') as h5file:
        mel_data = h5file.create_dataset(
            'mels',
            (n_files, n_mels, mel_len),
            maxshape=(n_files, n_mels, mel_len),
            chunks=(1, n_mels, mel_len))

        event_data = h5file.create_dataset(
            'events',
            (n_files, mel_len, n_labels),
            maxshape=(n_files, mel_len, n_labels),
            chunks=(1, mel_len, n_labels))

        fname_data = h5file.create_dataset(
            'fname',
            (n_files,),
            maxshape=(n_files,),
            chunks=(1,),
            dtype=h5py.string_dtype(encoding='utf-8'))

        idx = 0
        for fn in tqdm(audio_filenames):
            y, _ = librosa.load(fn, sr=sample_rate, mono=True)
            if len(y) / sample_rate < min_duration:
                continue

            m = librosa.feature.melspectrogram(
                y=y,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                n_mels=n_mels,
                fmin=f_min)

            if m.shape[1] > mel_len:
                m = m[:, :mel_len]

            yt_id = fn.stem[1:]
            mel_data[idx, :, :m.shape[1]] = m
            fname_data[idx] = yt_id

            for onset, offset, label in events[yt_id]:
                if label not in label_ind:
                    continue

                start = int(onset / hop_len_seconds)
                end = int(offset / hop_len_seconds) + 1
                event_data[idx, start:end, label_ind[label]] = 1

            assert np.sum(event_data[idx, :, :]) > 0
            idx += 1

        mel_data.resize((idx, n_mels, mel_len))
        event_data.resize((idx, mel_len, n_labels))
        fname_data.resize((idx,))


if __name__ == '__main__':
    main()
