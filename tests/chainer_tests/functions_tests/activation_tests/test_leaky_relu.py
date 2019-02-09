import random

import numpy

from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'slope': ['random', 0.0],
}))
@testing.fix_random()
@testing.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestLeakyReLU(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        if self.slope == 'random':
            self.slope = random.random()
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.leaky_relu(x, slope=self.slope),

    def forward_expected(self, inputs):
        x, = inputs
        y = utils.force_array(
            numpy.where(x >= 0, x, x * self.slope), self.dtype)
        return y,


testing.run_module(__name__, __file__)
