import numpy

from chainer import functions
from chainer import testing
from chainer.testing import backend


def _instance_normalization(x, gamma, beta, eps=2e-5):
    orig_shape = x.shape
    b, c = orig_shape[:2]
    x_reshaped = x.reshape((1, b * c) + orig_shape[2:])
    aggr_axes = (0,) + tuple(range(2, len(orig_shape)))
    expander = [Ellipsis] * len(orig_shape)
    for i in aggr_axes:
        expander[i] = None
    expander = tuple(expander)
    mean = numpy.mean(x_reshaped, axis=aggr_axes)
    var = numpy.var(x_reshaped, axis=aggr_axes)
    std = numpy.sqrt(var + eps)
    x_reshaped_normalized = (x_reshaped - mean[expander]) / std[expander]
    x_normalized = x_reshaped_normalized.reshape(orig_shape)
    y = gamma[expander] * x_normalized + beta[expander]
    return y


def _fixed_instance_normalization(
        x, gamma, beta, mean, var, eps=2e-5, decay=0.9):
    orig_shape = x.shape
    aggr_axes = (0,) + tuple(range(2, len(orig_shape)))
    expander = [Ellipsis] * len(orig_shape)
    for i in aggr_axes:
        expander[i] = None
    expander = tuple(expander)
    x_normalized = (x - mean[expander]) / numpy.sqrt(var[expander] + eps)
    y = gamma[expander] * x_normalized + beta[expander]
    return y


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'eps': [2e-5],
    'moving_avg': [True, False],
})))
@backend.inject_backend_tests(
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
        'cudnn_fast_batch_normalization': [True, False],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ])
class TestInstanceNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})
        self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
        self.check_double_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 1e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 1e-2})

        if self.moving_avg:
            channels = self.shape[1]
            self.running_mean = numpy.zeros(channels, self.dtype)
            self.running_var = numpy.ones(channels, self.dtype)
        else:
            self.running_mean, self.running_var = None, None

    def generate_inputs(self):
        shape, dtype = self.shape, self.dtype
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gamma = numpy.random.uniform(0.5, 1, shape[1]).astype(dtype)
        beta = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        return x, gamma, beta

    def forward(self, inputs, device):
        x, gamma, beta = inputs
        mean, var = self.running_mean, self.running_var
        if mean is not None:
            mean = device.send_array(mean)
            var = device.send_array(var)
        return functions.instance_normalization(
            x, gamma, beta, eps=self.eps, running_mean=mean, running_var=var),

    def forward_expected(self, inputs):
        x, gamma, beta = inputs
        y = _instance_normalization(x, gamma, beta, eps=self.eps)
        return y,


@testing.parameterize(*(testing.product({
    'shape': [(1, 4, 5, 5), (5, 4, 15)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': ['C', None],
})))
@backend.inject_backend_tests(
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
        'cudnn_fast_batch_normalization': [True, False],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ])
class TestFixedInstanceNormalization(testing.FunctionTestCase):

    def setUp(self):
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def generate_inputs(self):
        shape, dtype = self.shape, self.dtype
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gamma = numpy.random.uniform(0, 1, shape[1]).astype(dtype)
        beta = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        mean = numpy.random.uniform(-1, 1, shape[1]).astype(dtype)
        var = numpy.random.uniform(.5, 1, shape[1]).astype(dtype)
        return x, gamma, beta, mean, var

    def forward(self, inputs, device):
        x, gamma, beta, mean, var = inputs
        return functions.fixed_instance_normalization(
            x, gamma, beta, mean, var),

    def forward_expected(self, inputs):
        x, gamma, beta, mean, var = inputs
        y = _fixed_instance_normalization(x, gamma, beta, mean, var)
        return y,


testing.run_module(__name__, __file__)
