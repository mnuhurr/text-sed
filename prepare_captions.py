import csv

from pathlib import Path

from ac_data import read_captions
from common import read_yaml, write_json
from utils import load_tokenizer, data_fnames


def read_synthetic_captions(filename: str | Path, filter_ids: set[str] | None = None) -> dict[str, str]:
    captions = {}
    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # drop out the leading Y
            yt_id = row['id'][1:]
            if filter_ids is not None and yt_id not in filter_ids:
                continue

            caption = row['caption']
            captions[yt_id] = caption

    return captions


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)

    cache_dir = Path(cfg.get('cache_dir', 'cache'))

    tokenizer_dir = cfg.get('tokenizer_dir', 'tokenizer')
    tokenizer = load_tokenizer(tokenizer_dir)

    soundve_dir = Path(cfg.get('soundve_dir'))

    print('reading synthetic captions')
    captions = read_synthetic_captions(soundve_dir / 'Sound-VECaps_audio.csv')

    # get all the filenames from the data files (these are used for all splits)
    fnames = data_fnames(cache_dir / 'train_data.hdf')
    fnames = set(fnames)

    print('filtering captions')
    captions = {fn: caption for fn, caption in captions.items() if fn in fnames}
    print(f'got {len(captions)} captions from Sound-VECaps')

    data = {fn: {'tokens': [tokenizer.encode(caption).ids]} for fn, caption in captions.items()}
    write_json(cache_dir / 'soundve-caps.json', data)

    # audiocaps
    ac_dir = Path(cfg.get('audiocaps_dir'))
    captions = read_captions(ac_dir / 'dataset' / 'train.csv')

    # filter
    captions = {fn: caption for fn, caption in captions.items() if fn in fnames}
    print(f'got {len(captions)} captions from audiocaps')

    # t.ids for t in tokenizer.encode_batch(ac_train_capts[yid])
    data = {fn: {'tokens': [t.ids for t in tokenizer.encode_batch(captlist)]} for fn, captlist in captions.items()}
    write_json(cache_dir / 'audiocaps.json', data)


if __name__ == '__main__':
    main()
