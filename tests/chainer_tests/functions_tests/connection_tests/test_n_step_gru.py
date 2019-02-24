import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing.function import FunctionTestError


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


def _to_gpu(x):
    if x is None:
        return None
    elif isinstance(x, list):
        return [_to_gpu(xi) for xi in x]
    else:
        return cuda.to_gpu(x)


def _wrap_variable(x):
    if isinstance(x, list):
        return [_wrap_variable(xi) for xi in x]
    else:
        return chainer.Variable(x)


def _gen_uniform(shape, dtype):
    return numpy.random.uniform(-1, 1, shape).astype(dtype)


class BaseTestNStepGRU(object):

    batches = [3, 2, 1]
    n_layers = 3
    in_size = 3
    out_size = 2
    dropout = 0.0

    def generate_inputs(self):
        xs = [
            _gen_uniform((b, self.in_size), self.x_dtype) for b in self.batches
        ]
        hx = _gen_uniform(self.h_shape, self.x_dtype)
        inputs = [hx] + xs
        return tuple(inputs)

    def generate_grad_outputs(self, outputs_template):
        dys = [
            _gen_uniform((b, self.out_size), self.x_dtype)
            for b in self.batches
        ]
        dhy = _gen_uniform(self.h_shape, self.x_dtype)
        return tuple([dhy] + dys)

    def _send_weights(self, arrays, device):
        if isinstance(device, chainer.testing.backend.BackendConfig):
            f = device.get_array
        else:
            f = device.send
        return [[f(a) for a in arr] for arr in arrays]

    def _forward_util(self, inputs, device):
        hx, *xs = inputs
        ws = self._send_weights(self.ws, device)
        bs = self._send_weights(self.bs, device)
        hy, ys = functions.n_step_gru(
            self.n_layers, self.dropout, hx, ws, bs, xs)
        return hy, ys

    def forward(self, inputs, device):
        hy, ys = self._forward_util(inputs, device)
        ys_concat = functions.concat(ys, 0)
        return hy, ys_concat

    def forward_expected(self, inputs):
        hx, *xs = inputs
        hx = numpy.copy(hx)
        ws, bs = self.ws, self.bs
        y_expected = []
        for ind in range(self.length):
            x = xs[ind]
            batch = x.shape[0]
            for layer in range(self.n_layers):
                w, b = ws[layer], bs[layer]
                h_prev = hx[layer, :batch]
                # GRU
                z = sigmoid(x.dot(w[1].T) + h_prev.dot(w[4].T) + b[1] + b[4])
                r = sigmoid(x.dot(w[0].T) + h_prev.dot(w[3].T) + b[0] + b[3])
                h_bar = numpy.tanh(
                    x.dot(w[2].T) + r * ((h_prev).dot(w[5].T) + b[5]) + b[2])
                e_h = (1 - z) * h_bar + z * h_prev
                hx[layer, :batch] = e_h
                x = e_h
            y_expected.append(x)
        y_concat = numpy.concatenate(y_expected, 0)
        return hx, y_concat

    def run_test_backward(self, backend_config):
        # Runs the backward test.
        if self.skip_backward_test:
            raise unittest.SkipTest('skip_backward_test is set')

        # avoid cyclic import
        from chainer import gradient_check

        self.backend_config = backend_config
        self.before_test('test_backward')

        def f(*args):
            hy, ys = self._forward_util(args, backend_config)
            return (hy,) + ys

        def do_check():
            inputs = self._generate_inputs()
            outputs = self._forward_expected(inputs)
            grad_outputs = self._generate_grad_outputs(outputs)

            inputs = backend_config.get_array(inputs)
            grad_outputs = backend_config.get_array(grad_outputs)
            inputs = self._to_noncontiguous_as_needed(inputs)
            grad_outputs = self._to_noncontiguous_as_needed(grad_outputs)
            dhy, *dys = grad_outputs

            with FunctionTestError.raise_if_fail(
                    'backward is not implemented correctly'):
                gradient_check.check_backward(
                    f, inputs, grad_outputs, dtype=numpy.float64,
                    detect_nondifferentiable=self.dodge_nondifferentiable,
                    **self.check_backward_options)

        do_check()


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}))
@testing.inject_backend_tests(
    ['test_forward', 'test_backward'],
    # CPU tests
    [{}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
)
class TestNStepGRU(testing.FunctionTestCase, BaseTestNStepGRU):

    skip_double_backward_test = True

    def setUp(self):
        self.length = len(self.batches)
        self.h_shape = (self.n_layers, self.batches[0], self.out_size)
        self._prepare_weights()
        self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-4})
        self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-3})

    def _prepare_weights(self):
        out_size, W_dtype, x_dtype = self.out_size, self.W_dtype, self.x_dtype
        ws, bs = [], []
        for i in range(self.n_layers):
            weights, biases = [], []
            for j in range(6):
                if i == 0 and j < 3:
                    w_in = self.in_size
                else:
                    w_in = self.out_size
                weights.append(_gen_uniform((out_size, w_in), W_dtype))
                biases.append(_gen_uniform((out_size,), x_dtype))
            ws.append(weights)
            bs.append(biases)
        self.ws, self.bs = ws, bs


@testing.parameterize(*testing.product({
    'x_dtype': [numpy.float32],
    'W_dtype': [numpy.float32],
}))
@testing.inject_backend_tests(
    ['test_forward', 'test_backward'],
    # CPU tests
    [{}]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
)
class TestNStepBiGRU(testing.FunctionTestCase, BaseTestNStepGRU):

    def setUp(self):
        self.length = len(self.batches)
        self.h_shape = (self.n_layers * 2, self.batches[0], self.out_size)
        self._prepare_weights()
        self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-4})
        self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-3})

    def _prepare_weights(self):
        out_size, W_dtype, x_dtype = self.out_size, self.W_dtype, self.x_dtype
        ws, bs = [], []
        for i in range(self.n_layers):
            weights, biases = [], []
            for j in range(6):
                if i == 0 and j < 3:
                    w_in = self.in_size
                elif i > 0 and j < 3:
                    w_in = self.out_size * 2
                else:
                    w_in = self.out_size
                weights.append(_gen_uniform((out_size, w_in), W_dtype))
                biases.append(_gen_uniform((out_size,), x_dtype))
            ws.append(weights)
            bs.append(biases)
        self.ws, self.bs = ws, bs


testing.run_module(__name__, __file__)
