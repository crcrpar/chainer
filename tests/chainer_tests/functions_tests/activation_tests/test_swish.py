import numpy

from chainer import functions
from chainer import testing
from chainer import utils


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


@testing.parameterize(*testing.product_dict(
    [
        {'x_shape': (4, 3, 2), 'beta_shape': (3,),
         'extended_beta_shape': (1, 3, 1)},
        {'x_shape': (4, 3, 2), 'beta_shape': (3, 2),
         'extended_beta_shape': (1, 3, 2)},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
@testing.fix_random()
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
        'chainerx_device': ['native:0', 'cuda:1', 'cuda:1'],
    })
)
class TestSwish(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        beta = numpy.random.uniform(-1, 1, self.beta_shape).astype(self.dtype)
        return x, beta

    def forward(self, inputs, device):
        x, beta = inputs
        return functions.swish(x, beta),

    def forward_expected(self, inputs):
        x, beta = inputs
        beta = numpy.broadcast_to(
            numpy.reshape(beta, self.extended_beta_shape),
            x.shape)
        y = x * _sigmoid(beta * x)
        return utils.force_array(y, dtype=self.dtype),


testing.run_module(__name__, __file__)
