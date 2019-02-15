import six

import numpy

from chainer import functions
from chainer import testing


def _simple_group_normalization(x, groups, gamma, beta, eps=1e-5):
    batch_size, channels = x.shape[:2]
    x_reshape = x.reshape(batch_size, groups, channels // groups, -1)

    mu = numpy.mean(x_reshape, axis=(2, 3), keepdims=True)
    sigma = numpy.std(x_reshape, axis=(2, 3), keepdims=True)

    x_reshape = (x_reshape - mu) / (sigma + eps)
    x = x_reshape.reshape(x.shape)

    for i in six.moves.xrange(x.ndim):
        if i != 1:  # except for channel dim
            gamma = numpy.expand_dims(gamma, i)
            beta = numpy.expand_dims(beta, i)

    return x * gamma + beta


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'groups': [1, 2, 4],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
@testing.inject_backend_tests(
    None,
    [{}]
    +
    testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never'],
        'cuda_device': [0, 1],
    })
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['always'],
        'cudnn_fast_batch_normalization': [True, False],
        'cuda_device': [0, 1],
    })
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestGroupNormalization(testing.FunctionTestCase):

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gamma = numpy.random.uniform(-1, 1, self.shape[1]).astype(self.dtype)
        beta = numpy.random.uniform(-1, 1, self.shape[1]).astype(self.dtype)
        return x, gamma, beta

    def setUp(self):
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def forward(self, inputs, device):
        x, gamma, beta = inputs
        return functions.group_normalization(
            x, self.groups, gamma, beta, self.eps),

    def forward_expected(self, inputs):
        x, gamma, beta = inputs
        y = _simple_group_normalization(x, self.groups, gamma, beta, self.eps)
        return y,


testing.run_module(__name__, __file__)
