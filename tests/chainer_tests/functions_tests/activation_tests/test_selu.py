import random

import numpy

from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {}
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestSELU(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        self.alpha = random.random()
        self.scale = random.random()
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.selu(x, alpha=self.alpha, scale=self.scale),

    def forward_expected(self, inputs):
        x, = inputs
        y = numpy.where(x > 0, x, self.alpha * (numpy.exp(x) - 1))
        return utils.force_array(y * self.scale, dtype=self.dtype),


testing.run_module(__name__, __file__)
