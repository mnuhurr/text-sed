from pathlib import Path
import tokenizers
from tokenizers.normalizers import BertNormalizer

from common import read_yaml, init_log


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('fit-tokenizer', level=cfg.get('log_level', 'info'))

    captions = []

    for capt_fn in cfg.get('tokenizer_fit_files', []):
        captions.extend(Path(capt_fn).read_text().splitlines())

    logger.info(f'using {len(captions)} captions in total')

    vocab_size = cfg.get('vocab_size', 10000)
    min_frequency = cfg.get('min_frequency', 2)
    logger.info(f'vocab_size={vocab_size}, min_frequency={min_frequency}')

    special_tokens = ['<s>', '</s>', '<pad>', '<unk>', '<mask>']

    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer._tokenizer.normalizer = BertNormalizer(strip_accents=False, lowercase=True)
    tokenizer.train_from_iterator(
        captions,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens)

    tokenizer_dir = Path(cfg.get('tokenizer_dir', '.'))
    tokenizer_dir.mkdir(exist_ok=True, parents=True)
    tokenizer.save_model(str(tokenizer_dir), 'tokenizer')


if __name__ == '__main__':
    main()
