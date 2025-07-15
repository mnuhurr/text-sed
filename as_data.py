import csv

from pathlib import Path
from tqdm import tqdm


def read_labels(filename: str | Path) -> dict[str, str]:
    labels = {}
    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            labels[row['id']] = row['label']

    return labels


def add_events_to_list(event_list: list[tuple[float, float, str]], event: tuple[float, float, str]):
    # combine overlapping events
    for k in range(len(event_list)):
        # check that we have the same event class
        if event_list[k][2] != event[2]:
            continue

        # check if the new event is already included in the existing event
        if event_list[k][0] <= event[0] and event_list[k][1] >= event[1]:
            return

        if event_list[k][0] <= event[0] <= event_list[k][1]:
            new_event = (event_list[k][0], max(event_list[k][1], event[1]), event_list[k][2])
            event_list[k] = new_event
            return

        if event[0] <= event_list[k][0] <= event[1]:
            new_event = (min(event[0], event_list[k][0]), event_list[k][1], event_list[k][2])
            event_list[k] = new_event
            return

    event_list.append(event)


def read_events(filename: str | Path, drop_time: bool = False) -> tuple[dict[str, list[tuple[float, float, str]]], set[str]]:
    events = {}
    labels = set()
    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            fn = row['filename']
            onset = float(row['onset'])
            offset = float(row['offset'])
            label = row['event_label']

            if drop_time:
                fn = fn[:11]

            labels.add(label)

            event = (onset, offset, label)
            if fn not in events:
                events[fn] = [event]
            else:
                #events[fn].append(event)
                add_events_to_list(events[fn], event)

    return events, labels


def read_tags(filename: str | Path, drop_time: bool = False) -> tuple[dict[str, list[tuple[float, float, str]]], set[str]]:
    tags = {}
    labels = set()
    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            fn = row['filename']
            label = row['event_label']

            if drop_time:
                fn = fn[:11]

            labels.add(label)

            if fn not in tags:
                tags[fn] = {label}
            else:
                tags[fn].add(label)

    return tags, labels


def read_ac_captions(filename: str | Path) -> dict[str, list[str]]:
    captions = {}
    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f)

        for row in reader:
            yt_id = row['youtube_id']
            caption = row['caption']
            onset = int(row['start_time'])

            fn_key = f'{yt_id}_{1000 * onset}'
            if fn_key not in captions:
                captions[fn_key] = [caption]
            else:
                captions[fn_key].append(captions)

    return captions


def collect_audio_filenames(audio_dir: str | Path, filter_set: set[str] | list[str] | None = None) -> list[Path]:
    filenames = []

    filter_set = set(filter_set) if filter_set is not None else None

    for subdir in tqdm([sd for sd in Path(audio_dir).iterdir() if sd.is_dir()]):
        if not subdir.name.startswith('unbalanced'):
            continue

        fns = list(subdir.glob('*.wav'))
        if filter_set is not None:
            fns = [fn for fn in fns if fn.stem[1:] in filter_set]
        filenames.extend(fns)

    return filenames


def _test():
    fn = '/home/mnu/Documents/src/py/GoogleAudioSetReformatted/audioset_strong_train.tsv'
    tags, tl = read_tags(fn)
    as_yt_ids = [fn_key[:11] for fn_key in tags.keys()]

    fn = '/media/work/mnu/data/audiocaps/dataset/train.csv'
    caps = read_ac_captions(fn)
    ac_yt_ids = [fn_key[:11] for fn_key in caps.keys()]
    common_keys = set(caps.keys()).intersection(tags.keys())


if __name__ == '__main__':
    _test()
