import unittest

import numpy
import six
import warnings

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend


def _as_noncontiguous_array(array):
    # TODO(niboshi): cupy + cudnn test fails in F.fixed_batch_normalization.
    # Fix it and use testing.array._as_noncontiguous_array.
    def as_noncontiguous_array(arr):
        if arr is None:
            return None
        if isinstance(arr, (numpy.ndarray, cuda.ndarray)):
            xp = chainer.backend.get_array_module(arr)
            return xp.asfortranarray(arr)
        return testing.array._as_noncontiguous_array(arr)

    if isinstance(array, (list, tuple)):
        return type(array)([as_noncontiguous_array(arr) for arr in array])
    return as_noncontiguous_array(array)


def _batch_normalization(
        inputs, running_mean=None, running_var=None, decay=None):
    x, gamma, beta, mean, var, eps, expander = inputs
    mean_expanded = mean[expander]
    std = numpy.sqrt(var + eps)[expander]
    y_expect = (gamma[expander] * (x - mean_expanded) / std + beta[expander])

    if running_mean is not None or running_var is not None:
        m = x.size // gamma.size
        adjust = m / max(m - 1., 1.)  # unbiased estimation
        if running_mean is not None:
            running_mean *= decay
            running_mean += (1 - decay) * mean
        if running_var is not None:
            running_var *= decay
            running_var += (1 - decay) * adjust * var

    return y_expect


@testing.parameterize(*(testing.product_dict(
    testing.product({
        'param_shape': [(3,), (3, 4), (3, 2, 3)],
        'ndim': [0, 1, 2],
    }) + [
        {'input_shape': (5, 4, 3, 2), 'axis': (0, 2, 3)},
        {'input_shape': (5, 4), 'axis': 0},
        {'input_shape': (5, 4, 3), 'axis': (0, 1)},
    ],
    testing.product({
        'dtype': [numpy.float32],
        'eps': [2e-5, 5e-1],
        'contiguous': ['C', None],
        'running_statistics': [True, False],
    }),
) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'contiguous': ['C', None],
    'running_statistics': [True, False],
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
class TestBatchNormalization(testing.FunctionTestCase):

    # FIXME(crcrpar): chainerx, native:0, double_backward

    def setUp(self):
        self.decay = 0.9
        self.bn_options = {
            'decay': self.decay,
            'eps': self.eps,
        }
        if hasattr(self, 'axis'):
            self.bn_options['axis'] = self.axis
        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_double_backward_options = {
            'atol': 1e-3, 'rtol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'atol': 1e-2, 'rtol': 1e-2}

    def generate_inputs(self):
        dtype = self.dtype

        if not hasattr(self, 'axis'):
            param_shape = self.param_shape
            ndim = self.ndim
            shape = (5,) + param_shape + (2,) * ndim
        else:
            aggr_axes = self.axis
            if isinstance(self.axis, int):
                aggr_axes = self.axis,
            param_shape = tuple(
                s
                for i, s in enumerate(self.input_shape)
                if i not in aggr_axes
            )
            shape = self.input_shape

        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)

        if self.running_statistics:
            self.running_mean = numpy.random.uniform(
                -1, 1, param_shape).astype(dtype)
            self.running_var = numpy.random.uniform(
                -1, 1, param_shape).astype(dtype)
        else:
            self.running_mean = None
            self.running_var = None

        if not hasattr(self, 'axis'):
            head_ndim = gamma.ndim + 1
            aggr_axes = (0,) + tuple(six.moves.range(head_ndim, x.ndim))

            self.expander = (None, Ellipsis) + (None,) * ndim
        else:
            self.expander = tuple(
                None if i in aggr_axes else slice(None)
                for i in range(x.ndim)
            )

        mean = x.mean(axis=aggr_axes)
        var = x.var(axis=aggr_axes)
        self.mean = mean
        self.var = var
        return x, gamma, beta

    def forward(self, inputs, device):
        x, gamma, beta = inputs
        running_mean = device.send_array(self.running_mean)
        running_var = device.send_array(self.running_var)

        y = functions.batch_normalization(
            x, gamma, beta, running_mean=running_mean,
            running_var=running_var, **self.bn_options
        )
        return y,

    def forward_expected(self, inputs):
        if self.running_statistics:
            running_mean_expected = self.running_mean.copy()
            running_var_expected = self.running_var.copy()
        else:
            running_mean_expected = None
            running_var_expected = None
        y_expect = _batch_normalization(
            inputs + (self.mean, self.var, self.eps, self.expander),
            running_mean_expected, running_var_expected, self.decay)
        return y_expect,


@testing.parameterize(*(testing.product({
    'param_shape': [(3,), (3, 4), (3, 2, 3)],
    'ndim': [0, 1, 2],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float32],
    'c_contiguous': [True, False],
}) + testing.product({
    'param_shape': [(3,)],
    'ndim': [1],
    'eps': [2e-5, 5e-1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
})))
@backend.inject_backend_tests(
    None,
    # CPU tests
    [{'use_cuda': False}]
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
class TestFixedBatchNormalization(testing.FunctionTestCase):

    # FIXME(crcrpar): forward, cudnn, fast bn, float32

    def setUp(self):
        self.decay = 0.0
        self.expander = (None, Ellipsis) + (None,) * self.ndim

        self.check_forward_options = {'atol': 1e-4, 'rtol': 1e-3}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def generate_inputs(self):
        param_shape = self.param_shape
        dtype = self.dtype
        ndim = self.ndim
        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        shape = (5,) + param_shape + (2,) * ndim
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        mean = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        var = numpy.random.uniform(0.5, 1, param_shape).astype(dtype)
        return x, gamma, beta, mean, var

    def forward(self, inputs, device):
        x, gamma, beta, mean, var = inputs
        y = functions.fixed_batch_normalization(
            x, gamma, beta, mean, var, eps=self.eps
        )
        return y,

    def forward_expected(self, inputs):
        y_expect = _batch_normalization(inputs + (self.eps, self.expander))
        return y_expect,


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'eps': [2e-5, 5e-1],
    # TODO(bkvogel): Check float16 support again in next cuDNN version.
    'dtype': [numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestBatchNormalizationCudnnCall(unittest.TestCase):

    def setUp(self):
        ndim = 0
        param_shape = (3,)
        self.gamma = cuda.cupy.random.uniform(.5, 1,
                                              param_shape).astype(self.dtype)
        self.beta = cuda.cupy.random.uniform(-1, 1,
                                             param_shape).astype(self.dtype)
        shape = (7,) + param_shape + (2,) * ndim
        self.x = cuda.cupy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, shape).astype(self.dtype)
        self.args = [self.x, self.gamma, self.beta]
        head_ndim = self.gamma.ndim + 1
        self.aggr_axes = (0,) + tuple(six.moves.range(head_ndim, self.x.ndim))
        self.mean = self.x.mean(axis=self.aggr_axes)
        self.var = self.x.var(axis=self.aggr_axes) + self.eps
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('>=auto', 5000)

    def forward(self):
        return functions.batch_normalization(
            *[chainer.Variable(i) for i in self.args], eps=self.eps,
            running_mean=self.mean, running_var=self.var)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch(
                    'cupy.cudnn.batch_normalization_forward_training'
            ) as func:
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            with testing.patch(
                    'cupy.cudnn.batch_normalization_backward'
            ) as func:
                y.backward()
                self.assertEqual(func.called, self.expect)


@attr.cudnn
class TestBatchNormalizationCudnnEps(unittest.TestCase):
    def setUp(self):
        ndim = 0
        param_shape = (3,)
        dtype = numpy.float32
        gamma = cuda.cupy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = cuda.cupy.random.uniform(-1, 1, param_shape).astype(dtype)
        shape = (7,) + param_shape + (2,) * ndim
        x = cuda.cupy.random.uniform(-1, 1, shape).astype(dtype)
        self.args = [x, gamma, beta]

    def test_valid(self):
        functions.batch_normalization(*self.args, eps=1e-5)

    def test_invalid(self):
        with self.assertRaises(RuntimeError):
            functions.batch_normalization(*self.args, eps=2e-6)


@attr.cudnn
class TestFixedBatchNormalizationCudnnEps(unittest.TestCase):
    def setUp(self):
        ndim = 0
        param_shape = (3,)
        dtype = numpy.float32
        gamma = cuda.cupy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = cuda.cupy.random.uniform(-1, 1, param_shape).astype(dtype)
        mean = cuda.cupy.random.uniform(-1, 1, param_shape).astype(dtype)
        var = cuda.cupy.random.uniform(-1, 1, param_shape).astype(dtype)
        shape = (7,) + param_shape + (2,) * ndim
        x = cuda.cupy.random.uniform(-1, 1, shape).astype(dtype)
        self.args = [x, gamma, beta, mean, var]

    def test_valid(self):
        functions.fixed_batch_normalization(*self.args, eps=1e-5)

    def test_invalid(self):
        with self.assertRaises(RuntimeError):
            functions.fixed_batch_normalization(*self.args, eps=2e-6)


class TestBatchNormalizationWarning(unittest.TestCase):
    def setUp(self):
        pass

    def create_batch(self, param_shape, x_shape):
        dtype = numpy.float32
        gamma = numpy.random.uniform(.5, 1, param_shape).astype(dtype)
        beta = numpy.random.uniform(-1, 1, param_shape).astype(dtype)
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        args = [x, gamma, beta]
        return args

    def test_invalid_batch(self):
        args = self.create_batch((3,), (1, 3))
        with testing.assert_warns(UserWarning):
            functions.batch_normalization(*args)

    def test_invalid_batch_no_batch_axis(self):
        args = self.create_batch((1, 3,), (1, 3, 1))
        with testing.assert_warns(UserWarning):
            functions.batch_normalization(*args, axis=2)

    def test_valid_batch(self):
        args = self.create_batch((3,), (1, 3, 2, 2))
        with warnings.catch_warnings(record=True) as w:
            functions.batch_normalization(*args)
            assert len(w) == 0

    def test_valid_batch_no_batch_axis(self):
        args = self.create_batch((1, 3,), (1, 3, 2))
        with warnings.catch_warnings(record=True) as w:
            functions.batch_normalization(*args, axis=2)
            assert len(w) == 0


testing.run_module(__name__, __file__)
