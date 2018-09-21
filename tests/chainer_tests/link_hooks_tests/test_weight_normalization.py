import unittest

import numpy as np

import chainer
from chainer.backends import cuda
from chainer import links
from chainer import link_hooks
from chainer import testing
from chainer.testing import attr


@testing.product({
    'link': [links.Convolution2D, links.Deconvolution2D, links.Linear]
})
class TestWeightNormalization(unittest.TestCase):
