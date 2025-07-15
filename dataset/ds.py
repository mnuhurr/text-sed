from functools import reduce
from pathlib import Path

import h5py
import numpy as np
import torch
from tokenizers.implementations import ByteLevelBPETokenizer


def collate_tokens(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    bs = len(batch)
    max_len = max(map(len, batch))

    x = torch.zeros(bs, max_len, dtype=torch.int64)
    mask = torch.ones(bs, max_len, dtype=bool)

    for k, tokens in enumerate(batch):
        tlen = len(tokens)
        x[k, :tlen] = tokens
        mask[k, :tlen] = False

    return x, mask


def collate_tagging(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, tags = list(zip(*batch))
    x, x_mask = collate_tokens(tokens)
    y = torch.stack(tags)

    return x, x_mask, y


def collate_multimodal(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # assume audio is the same length, so only collate the text tokens
    mels, tokens, events = list(zip(*batch))
    x = torch.stack(mels)
    y, y_mask = collate_tokens(tokens)
    z = torch.stack(events)
    return x, y, y_mask, z


def collate_multimodal_test(batch):
    mels, tokens, events, fnames = list(zip(*batch))
    x = torch.stack(mels)
    y, y_mask = collate_tokens(tokens)
    z = torch.stack(events)
    return x, y, y_mask, z, fnames

def collate_multimodal_tag(batch):
    mels, tokens, events, use_weak = list(zip(*batch))
    x = torch.stack(mels)
    y, y_mask = collate_tokens(tokens)
    z = torch.stack(events)
    use_weak = torch.as_tensor(use_weak)
    return x, y, y_mask, z, use_weak


def collate_multimodal_incorrect(batch):
    mels, tokens, events, fnames, correct = list(zip(*batch))
    x = torch.stack(mels)
    y, y_mask = collate_tokens(tokens)
    z = torch.stack(events)
    w = torch.stack(correct)
    return x, y, y_mask, z, fnames, w


class SEDDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_filename: str | Path,
                 file_idx: list[int] | None = None,
                 downsampling_factor: int = 1,
                 downsampling_ceil_mode: bool = False,
                 slice_swapping: bool = False):

        self.data = h5py.File(data_filename, 'r')
        self.file_idx = file_idx if file_idx is not None else list(range(self.data['mels'].shape[0]))
        self.downsampling_factor = downsampling_factor
        self.slice_swapping = slice_swapping
        self.downsampling_ceil_mode = downsampling_ceil_mode

    def __len__(self) -> int:
        return len(self.file_idx)

    def _logmels(self, idx: int) -> torch.Tensor:
        x = torch.as_tensor(self.data['mels'][idx])
        x = torch.log(x + torch.finfo(x.dtype).eps)
        return x

    def _events(self, idx: int) -> torch.Tensor:
        y = torch.as_tensor(self.data['events'][idx])

        if self.downsampling_factor > 1:
            y = torch.nn.functional.avg_pool1d(y.T, self.downsampling_factor, ceil_mode=self.downsampling_ceil_mode).T

        return y

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.file_idx[index]

        x = self._logmels(item)
        y = self._events(item)

        if self.slice_swapping and np.random.random() < 0.5:
            # take the slicing position so that it doesn't cut any event into two
            no_event_idx = list(torch.where(torch.all(y == 0, dim=1))[0])
            possible = [k for k in no_event_idx if k - 1 in no_event_idx]
            if len(possible) > 0:
                y_m = np.random.choice(possible)
                x_m = self.downsampling_factor * y_m

                xs = torch.zeros_like(x)
                xs[:, :x_m] = x[:, -x_m:]
                xs[:, x_m:] = x[:, :-x_m]

                ys = torch.zeros_like(y)
                ys[:y_m] = y[-y_m:]
                ys[y_m:] = y[:-y_m]

        return x, y

    @property
    def n_labels(self) -> int:
        return self.data['events'].shape[-1]


class SEDCaptionDataset(SEDDataset):
    def __init__(self, *args, **kwargs):
        self.tokens = kwargs.pop('tokens')
        super().__init__(*args, **kwargs)

    def _tokens(self, key: str) -> torch.Tensor:
        if key not in self.tokens:
            return torch.as_tensor([], dtype=torch.int64)

        if len(self.tokens[key]) > 1:
            c_idx = torch.randint(len(self.tokens[key]), size=())
            tokens = self.tokens[key][c_idx]
        else:
            tokens = self.tokens[key][0]
        return torch.as_tensor(tokens, dtype=torch.int64)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, z = super().__getitem__(index)
        item = self.file_idx[index]

        fname = self.data['fname'][item].decode('utf-8')
        y = self._tokens(fname)
        return x, y, z


class SEDCaptionTestDataset(SEDCaptionDataset):
    def __getitem__(self, index: int):
        x, y, z = super().__getitem__(index)

        item = self.file_idx[index]
        fname = self.data['fname'][item].decode('utf-8')

        return x, y, z, fname


class SEDCaptionTagDataset(SEDCaptionDataset):
    def __init__(self, *args, **kwargs):
        self.weak_idx = set(kwargs.pop('weak_idx', []))
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        x, y, z = super().__getitem__(index)
        w = self.file_idx[index] in self.weak_idx
        return x, y, z, w


def _collect_labels(events: dict[str, list[tuple[float, float, str]]]) -> set[str]:
    labels = set()

    for event_list in events.values():
        labels.update([lab for _, _, lab in event_list])

    return labels


class SEDTestDataset(SEDDataset):
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(index)

        item = self.file_idx[index]
        fname = self.data['fname'][item].decode('utf-8')
        return x, y, fname


class CaptionTagDataset(torch.utils.data.Dataset):
    def __init__(self,
                 captions: dict[str, list[list[int]]],
                 tags: dict[str, list[str]],
                 keys: list[str] | None = None,
                 labels: list[str] | None = None):

        if labels is None:
            self.labels = sorted(reduce(lambda x, y: set(x).union(y), tags.values(), set()))
        else:
            self.labels = labels
        self.label_ind = {label: k for k, label in enumerate(self.labels)}

        self.captions = captions
        self.tags = tags
        self.keys = keys if keys is not None else sorted(set(captions.keys()).intersection(tags.keys()))

    def __len__(self) -> int:
        return len(self.keys)

    def _caption(self, key: str) -> torch.Tensor:
        if len(self.captions[key]) > 1:
            c_idx = torch.randint(len(self.captions[key]), size=())
            caption = self.captions[key][c_idx]
        else:
            caption = self.captions[key][0]
        return torch.as_tensor(caption, dtype=torch.int64)

    def _tags(self, key: str) -> torch.Tensor:
        tag_vec = torch.zeros(len(self.labels), dtype=torch.float32)

        for tag in self.tags[key]:
            if tag not in self.label_ind:
                continue

            tag_vec[self.label_ind[tag]] = 1

        return tag_vec

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        fn = self.keys[item]

        x = self._caption(fn)
        y = self._tags(fn)

        return x, y


def _collect_labels(tags: dict[str, list[str]]) -> set[str]:
    all_labels = set()

    for file_tags in tags.values():
        all_labels.update(file_tags)

    return all_labels


class RandomTagDataset(SEDDataset):
    def __init__(self,
                 data_filename: str | Path,
                 tags: dict[str, list[str]],
                 class_names: dict[str, str],
                 tokenizer: ByteLevelBPETokenizer,
                 n_incorrect: int = 0,
                 file_idx: list[int] | None = None,
                 downsampling_factor: int = 1,
                 labels: list[str] | None = None,
                 ignore_labels: list[str] | set[str] | None = None,
                 start_token: int = 0,
                 end_token: int = 1):

        super().__init__(data_filename=data_filename, file_idx=file_idx, downsampling_factor=downsampling_factor)

        self.labels = set(labels) if labels is not None else _collect_labels(tags)
        #self.class_names = {c: tokenizer.encode(class_names[c]).ids for c in class_names}
        self.class_names = class_names
        self.tokenizer = tokenizer
        self.tags = tags
        self.n_incorrect = n_incorrect
        self.start_token = start_token
        self.end_token = end_token
        self.ignore_labels = set(ignore_labels) if ignore_labels is not None else set()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        x, y = super().__getitem__(index)

        item = self.file_idx[index]
        fname = self.data['fname'][item].decode('utf-8')
        z_correct = [t for t in self.tags[fname] if t in self.labels and t not in self.ignore_labels]

        # add incorrect tags
        z_incorrect = []
        if self.n_incorrect > 0:
            all_incorrect = sorted(self.labels - set(z_correct))
            incorrect = np.random.choice(all_incorrect, self.n_incorrect, replace=False)
            z_incorrect.extend(list(incorrect))

        zt = sorted(z_correct + z_incorrect)
        # order zt?
        np.random.shuffle(zt)

        # take the label names, concatenate and tokenize. the comma is important for the text encoder
        zt = ', '.join([self.class_names[t] for t in zt])
        z = self.tokenizer.encode(zt).ids

        #z = reduce(lambda x, y: x + y, zt, [])
        #z = [self.start_token] + z + [self.end_token]

        z = torch.as_tensor(z, dtype=torch.int64)

        return x, z, y, fname


class IncorrectTagDataset(SEDDataset):
    def __init__(self,
                 data_filename: str | Path,
                 tags: dict[str, list[str]],
                 class_names: dict[str, str],
                 tokenizer: ByteLevelBPETokenizer,
                 inspect_label: str,
                 no_errors: bool = False,
                 file_idx: list[int] | None = None,
                 downsampling_factor: int = 1,
                 labels: list[str] | None = None,
                 start_token: int = 0,
                 end_token: int = 1):

        super().__init__(data_filename=data_filename, file_idx=file_idx, downsampling_factor=downsampling_factor)

        self.labels = set(labels) if labels is not None else _collect_labels(tags)
        self.class_names = class_names
        self.tokenizer = tokenizer
        self.inspect_label = inspect_label
        self.no_errors = no_errors
        self.tags = tags
        self.start_token = start_token
        self.end_token = end_token

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]:
        x, y = super().__getitem__(index)

        item = self.file_idx[index]
        fname = self.data['fname'][item].decode('utf-8')
        z = [t for t in self.tags[fname] if t in self.labels]

        # allow none for baseline
        if self.inspect_label in z:
            if not self.no_errors:
                z.remove(self.inspect_label)
            had_label = 1
        else:
            if not self.no_errors:
                z.append(self.inspect_label)
            had_label = 0

        z = sorted(z)
        np.random.shuffle(z)

        # take the label names, concatenate and tokenize. the comma is important for the text encoder
        z = ', '.join([self.class_names[t] for t in z])
        z = self.tokenizer.encode(z).ids

        z = torch.as_tensor(z, dtype=torch.int64)
        w = torch.as_tensor(had_label, dtype=torch.int64)

        return x, z, y, fname, w


