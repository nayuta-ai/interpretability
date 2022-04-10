import copy
import os

from config.parse_args import parse_args
from config.defaults import _C


def parse_yacs():
    cfg = _C.clone()
    cfg.freeze()
    return cfg


__all__ = ['parse_args', 'parse_yacs']