import os
import tempfile
import unittest

import numpy
import pytest

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import links
from chainer.serializers import npz
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import type_check


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64, chainer.mixed16],
    'nobias': [True, False],
}))
@testing.inject_backend_tests(
    None,
    [{}]
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestLinear(testing.LinkTestCase):

    param_names = ('W', 'b')

    def generate_params(self):
        initialW = numpy.random.uniform(
            -1, 1, (3, 2)).astype(self.dtype)
        initial_bias = numpy.random.uniform(
            -1, 1, (3,)).astype(self.dtype)
        print('generate_params', initialW.dtype, initial_bias.dtype)
        return initialW, initial_bias

    def generate_inputs(self):
        x = numpy.random.uniform(
            -1, 1, (1, 2)).astype(self.dtype)
        return x,

    def create_link(self, initializers):
        initialW, initial_bias = initializers
        link = chainer.links.Linear(
            2, 3, initialW=initialW, initial_bias=initial_bias,
            nobias=self.nobias)
        if self.nobias:
            print('create_link', link.W.dtype)
        else:
            print('create_link', link.W.dtype, link.b.dtype)

        return link

    def forward(self, link, inputs, device):
        x, = inputs
        return link(x),

    def forward_expected(self, link, inputs):
        W = link.W.array
        if self.nobias:
            b = 0
        else:
            b = link.b.array
        x, = inputs
        expected = x.dot(W.T) + b
        return expected,


@testing.parameterize(*testing.product({
    'input_variable': [True, False],
    'linear_args': [(None, 2), (2,)],
}))
class TestLinearParameterShapePlaceholder(unittest.TestCase):

    in_size = 3
    in_shape = (in_size,)

    def setUp(self):
        self.link = links.Linear(*self.linear_args)
        self.out_size = self.linear_args[-1]
        temp_x = numpy.random.uniform(
            -1, 1, (self.out_size, self.in_size)).astype(numpy.float32)
        if self.input_variable:
            self.link(chainer.Variable(temp_x))
        else:
            self.link(temp_x)
        W = self.link.W.array
        W[...] = numpy.random.uniform(-1, 1, W.shape)
        b = self.link.b.array
        b[...] = numpy.random.uniform(-1, 1, b.shape)
        self.link.cleargrads()

        x_shape = (4,) + self.in_shape
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, (4, self.out_size)).astype(numpy.float32)
        self.y = self.x.reshape(4, -1).dot(W.T) + b

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = self.link(x)
        assert y.dtype == numpy.float32
        testing.assert_allclose(self.y, y.array)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            self.link, x_data, y_grad, (self.link.W, self.link.b), eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.link.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_serialization(self):
        lin1 = links.Linear(self.out_size)
        x = chainer.Variable(self.x)
        # Must call the link to initialize weights.
        lin1(x)
        w1 = lin1.W.array
        fd, temp_file_path = tempfile.mkstemp()
        os.close(fd)
        npz.save_npz(temp_file_path, lin1)
        lin2 = links.Linear(self.out_size)
        npz.load_npz(temp_file_path, lin2)
        w2 = lin2.W.data
        self.assertEqual((w1 == w2).all(), True)


class TestEmptyBatchInitialize(unittest.TestCase):

    def setUp(self):
        self.link = links.Linear(4)
        self.x = numpy.random.uniform(-1, 1, (0, 3)).astype(numpy.float32)

    def test_empty_batch_dim(self):
        y = self.link(chainer.Variable(self.x))
        assert y.shape == (0, 4)


class TestInvalidLinear(unittest.TestCase):

    def setUp(self):
        self.link = links.Linear(3, 2)
        self.x = numpy.random.uniform(-1, 1, (4, 1, 2)).astype(numpy.float32)

    def test_invalid_size(self):
        with pytest.raises(type_check.InvalidType):
            self.link(chainer.Variable(self.x))


testing.run_module(__name__, __file__)
