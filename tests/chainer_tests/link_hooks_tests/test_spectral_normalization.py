import os
import tempfile
import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.links as L
from chainer.link_hooks.spectral_normalization import SpectralNormalization
from chainer import testing


'''TODO(crcrpar): In each link, check
    - existences of auxiliary variables & buffers
    - Training Forward & Backward
    - Testing Forward (& the output is the same as the latest result)
    - Serialization
Also, I have to check Deconv things.
'''
KSIZE, STRIDE, PAD = 3, 1, 1
H, W = 4, 4


@testing.parameterize(*testing.product({
    'use_gamma': [True, False],
    'late_init': [True, False],
    'in_size': [3],
    'out_size': [5],
}))
class TestSpectralNormalizationLinear(unittest.TestCase):
    def setUp(self):
        if self.late_init:
            init_args = (None, self.out_size)
        else:
            init_args = (self.in_size, self.out_size)
        self.init_args = init_args
        self.layer = L.Linear(*self.init_args)
        self.hook = SpectralNormalization(use_gamma=self.use_gamma)
        self.layer.add_hook(self.hook)
        self.x = numpy.random.uniform(-1, 1, (10, self.in_size)).astype(numpy.float32)

    def test_initialization(self):
        if not self.late_init:
            self.assertTrue(hasattr(self.link, 'W_pre'))
            self.assertTrue(hasattr(self.link, 'u'))
            if self.use_gamma:
                self.assertTrue(hasattr(self.link, 'gamma'))
        else:
            self.assertFalse(hasattr(self.link, 'W_pre'))
            self.assertFalse(hasattr(self.link, 'u'))
            if self.use_gamma:
                self.assertFalse(hasattr(self.link, 'gamma'))

    def test_serialization(self):
        y = self.layer(self.x)
        fd, temp_file_path = tempfile.mkstemp()
        os.close(fd)
        chainer.serializers.npz.save_npz(temp_file_path, self.layer)
        layer = L.Linear(*self.init_args)
        chainer.serializers.npz.load_npz(temp_file_path, layer)
        with chainer.using_config('train', False):
            y2 = layer(self.x)
        numpy.testing.assertEqual(y.array, y2.array)
        numpy.testing.assertEqual(self.layer.u, layer.u)

    def test_forward(self):
        pass
