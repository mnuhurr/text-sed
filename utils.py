import csv
import h5py
import numpy as np
import torch

from functools import reduce
from pathlib import Path

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import BertNormalizer

from common import read_json
from models.tf import MultimodalCRNNConfig, MultimodalCRNN


def load_checkpoint(ckpt_path: str | Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    config = ckpt['config']
    print(config)
    if isinstance(config, MultimodalCRNNConfig):
        model = MultimodalCRNN(config)

    model.load_state_dict(ckpt['state_dict'])
    return model


def data_sizes(filename: str | Path) -> tuple[int, int, int]:
    with h5py.File(filename, 'r') as h5_file:
        n_files, n_mels, _ = h5_file['mels'].shape
        n_labels = h5_file['events'].shape[-1]

    return n_files, n_mels, n_labels


def data_fnames(filename: str | Path) -> list[str]:
    with h5py.File(filename, 'r') as h5_file:
        fnames = h5_file['fname'][:]
        fnames = [fn.decode('utf-8') for fn in fnames]
        return fnames


def data_tag_idx(filename: str | Path,
                 tags: dict[str, list[str]],
                 file_idx: list[int] | None = None,
                 labels: list[str] | None = None) -> dict[str, list[int]]:
    with h5py.File(filename, 'r') as h5_file:
        fnames = h5_file['fname'][:]
        fnames = [fn.decode('utf-8') for fn in fnames]

    if labels is None:
        # no labels provided, collect all the labels from the data
        all_labels = reduce(lambda x, y: x.union(y), tags.values(), set())
        labels = sorted(all_labels)

    # make it a set to make the lookup a bit faster
    fn_idx = set(file_idx) if file_idx is not None else None
    idx = {label: [] for label in labels}
    for k, fname in enumerate(fnames):
        # filter if the file index was provided
        if fn_idx is not None and k not in fn_idx:
            continue

        for tag in tags[fname]:
            if tag in idx:
                idx[tag].append(k)

    return idx


def train_val_idx(n_idx: int, validation_size: float = 0.1, seed: int | None = None) -> tuple[list[int], list[int]]:
    idx = list(range(n_idx))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n_idx * validation_size)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def read_labels(filename: str | Path) -> tuple[list[str], dict[str, str]]:
    """read labels from the generated csv file"""
    labels = []
    names = {}
    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row['mid']
            name = row['label']
            labels.append(row['mid'])
            names[mid] = name

    # labels are oreder by the number of occurrences in the csv file, model/data uses alphabetical order
    return sorted(labels), names


def load_tokenizer(tokenizer_path, max_length=None):
    vocab_fn = str(Path(tokenizer_path) / 'tokenizer-vocab.json')
    merges_fn = str(Path(tokenizer_path) / 'tokenizer-merges.txt')

    tokenizer = ByteLevelBPETokenizer(vocab_fn, merges_fn)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ('</s>', tokenizer.token_to_id('</s>')),
        ('<s>', tokenizer.token_to_id('<s>')),
    )

    tokenizer._tokenizer.normalizer = BertNormalizer(strip_accents=False, lowercase=True)

    if max_length is not None:
        tokenizer.enable_truncation(max_length=max_length)

    return tokenizer


def read_tokens(combined_data_fn: str | Path) -> dict[str, list[list[int]]]:
    data = read_json(combined_data_fn)

    tokens = {fn: data[fn]['tokens'] for fn in data}
    return tokens


def get_mask_token_id(tokenizer_path: str | Path) -> int:
    tokenizer = load_tokenizer(tokenizer_path)
    return tokenizer.token_to_id('<mask>')


def extract_events(y_pred: np.ndarray, hop_length_seconds: float, threshold: float = 0.5) -> list[tuple[float, float, int]]:
    """input: (t x k), where t is the number of time steps and k is the number of classes"""
    timestamps = np.arange(y_pred.shape[0] + 1) * hop_length_seconds
    onset = timestamps[:-1]
    offset = timestamps[1:]

    ybin = (y_pred >= threshold).astype(np.int16)

    ydiff = np.diff(ybin, axis=0)

    events = []
    for k in range(ybin.shape[1]):
        # every class separately
        starts = np.where(ydiff[:, k] > 0)[0]
        ends = np.where(ydiff[:, k] < 0)[0]

        # handle the special case that we have an event immediately
        if ybin[0, k] == 1:
            starts = np.concatenate([[-1], starts])

        elif ybin[-1, k] == 1:
            ends = np.concatenate([ends, [len(offset) - 1]])

        if len(ends) < len(starts):
            ends = np.concatenate([ends, [len(offset) - 1]])

        assert len(starts) == len(ends)
        starts = starts + 1

        for s, e in zip(starts, ends):
            events.append((onset[s], offset[e], k))

    return events


def make_event_list(y_pred: torch.Tensor, labels: list[str], hop_length_seconds: float, threshold: float = 0.5):
    events = extract_events(y_pred.numpy(), hop_length_seconds, threshold)
    events = [(onset, offset, labels[k]) for onset, offset, k in events]

    return events


def median_pool(x: torch.Tensor,
                kernel_size: int | list[int] | tuple[int],
                stride: int | list[int] | tuple[int] = 1,
                padding: int | list[int] | tuple[int] = 0,
                pad_mode: str = 'constant',
                pad_constant: float = 0.0) -> torch.Tensor:

    kernel_size = torch.nn.modules.utils._pair(kernel_size)
    stride = torch.nn.modules.utils._pair(stride)
    padding = torch.nn.modules.utils._quadruple(padding)

    if pad_mode == 'reflect':
        x = torch.nn.functional.pad(x, padding, mode='reflect')
    elif pad_mode == 'constant':
        x = torch.nn.functional.pad(x, padding, mode='constant', value=pad_constant)
    else:
        raise ValueError(f'expected padding mode to be either \'constant\' or \'reflect\', got {pad_mode}')

    x = x.unfold(1, kernel_size[0], stride[0]).unfold(2, kernel_size[1], stride[1])
    x = x.contiguous().view(x.size()[:3] + (-1,)).median(dim=-1)[0]
    return x


def _test():
    #x = np.random.random(size=(10, 4))
    #print(extract_events(x, 0.5))
    x = torch.randn(1, 5, 2)
    print(x)

    x = median_pool(x, kernel_size=(3, 1), pad_mode='constant', pad_constant=-1e9, padding=(0, 0, 1, 1))
    print(x)
    print(x.shape)

if __name__ == '__main__':
    _test()
