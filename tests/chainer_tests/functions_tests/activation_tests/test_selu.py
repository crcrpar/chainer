import random

import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@testing.fix_random()
@testing.backend.inject_backend_tests(
    None,
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestSELU(testing.FunctionTestCase):

    def setUp(self):
        self.alpha = random.random()
        self.scale = random.random()
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def generate_inputs(self):
        # Avoid unstability of numeraical grad
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        x[(-0.01 < x) & (x < 0.01)] = 0.5
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.selu(x, alpha=self.alpha, scale=self.scale),

    def forward_expected(self, inputs):
        x, = inputs
        y = numpy.where(x >= 0, x, self.alpha * (numpy.exp(x) - 1))
        y *= self.scale
        return y.astype(self.dtype),


testing.run_module(__name__, __file__)
