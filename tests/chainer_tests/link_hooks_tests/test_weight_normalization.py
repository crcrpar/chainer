import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer import testing


class TestWeightNormalizationHook(unittest.TestCase):

    def setUp(self):
        pass


testing.run_module(__name__, __file__)
