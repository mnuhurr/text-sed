import csv

from pathlib import Path


def read_captions(filename: str | Path) -> dict[str, list[str]]:
    captions = {}
    with Path(filename).open('rt') as f:
        reader = csv.DictReader(f)

        for row in reader:
            yid = row['youtube_id']
            capt = row['caption']

            if yid not in captions:
                captions[yid] = [capt]
            else:
                captions[yid].append(capt)

    return captions
