import unittest

import numpy
import pytest

from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'shape': [(200, 3)],
    'reduce': ['mean', 'no'],
    'norm': ['L1', 'L2', 'Huber'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'label_dtype': [numpy.int32],
}))
@testing.inject_backend_tests(
    None,
    # CPU
    [{}]
    # GPU
    + testing.product({
        'use_cuda': [True],
    })
    # ChainerX
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestHingeMaxMargin(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 5e-2, 'rtol': 5e-2})
            self.check_double_backward_options.update(
                {'atol': 5e-2, 'rtol': 5e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        t = numpy.random.randint(
            0, self.shape[1], self.shape[0]).astype(self.label_dtype)
        return (x, t)

    def forward(self, inputs, backend):
        x, t = inputs
        return functions.hinge_max_margin(x, t, self.norm, self.reduce),

    def forward_expected(self, inputs):
        x, t = inputs
        mask = numpy.zeros_like(x)
        mask[:, t] = -1
        tmp = numpy.copy(x)
        tmp[:, t] = numpy.finfo(x.dtype).min
        mask[:, numpy.argmax(tmp, 1)] = 1
        margin = numpy.maximum(0, 1 + numpy.sum(mask * x, 1))

        if self.norm == 'L1':
            loss = margin
        elif self.norm == 'L2':
            loss = margin ** 2
        else:
            # norm is Huber
            quad = (margin < 2).astype(x.dtype)
            loss = margin ** 2 / 4 * quad + (margin - 1) * (1 - quad)
        if self.reduce == 'mean':
            loss = utils.force_array(numpy.mean(loss), dtype=self.dtype)
        return loss,


@testing.parameterize(*testing.product({
    'shape': [(10, 5)],
    'dtype': [numpy.float32],
    'label_dtype': [numpy.int32],
}))
@testing.inject_backend_tests(
    ['test_invalid_norm', 'test_invalid_reduce'],
    # CPU
    [{}]
    # GPU
    + testing.product({
        'use_cuda': [True],
    })
    # ChainerX
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestHingeMaxMarginInvalidOption(unittest.TestCase):

    def setUp(self):
        shape = (10, 5)
        self.x = numpy.random.uniform(
            -1, 1, shape).astype(self.dtype)
        self.t = numpy.random.randint(
            0, 5, 10).astype(self.label_dtype)

    def test_invalid_norm(self, backend_config):
        x, t = backend_config.get_array((self.x, self.t))
        with backend_config, pytest.raises(NotImplementedError):
            functions.hinge_max_margin(x, t, 'invalid_norm', 'mean')

    def test_invalid_reduce(self, backend_config):
        x, t = backend_config.get_array((self.x, self.t))
        with backend_config, pytest.raises(ValueError):
            functions.hinge_max_margin(x, t, 'L1', 'invalid_option')


testing.run_module(__name__, __file__)
