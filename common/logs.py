import logging
from pathlib import Path
from typing import Union


def init_log(name: str = __name__, filename: Union[str, Path] = None, level: Union[int, str] = 'info', stream: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)

    if type(level) == str:
        level = level.upper()

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    if filename is not None:
        filehandler = logging.FileHandler(filename)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

    if stream:
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        logger.addHandler(streamhandler)

    logger.propagate = False

    return logger
