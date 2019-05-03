import unittest

import numpy
import pytest

from chainer import functions
from chainer import links
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'embed_size': [],
    'n_head': [],
    'qsize': [],
    'ksize': [],
    'vsize': [],
    'attention_dropout': [],
    'post_dropout': [],
    'scaling': [None, 1.0],
    'initialW': [None],
    'initial_bias': [None],
    'nobias': [True, False],
    'nobias_kv': [True, False],
    'add_zero_attention': [True, False],
}))
class TestMultiHeadAttention(unittest.TestCase):

    def test_forward(self):
        pass

    def test_forward(self):
        pass
