import numpy

from chainer import functions
from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'reduce': ['mean', 'no'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
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
class TestBlackOut(testing.FunctionTestCase):

    def setUp(self):
        self.batch_size = 5
        self.in_size = 4
        self.n_vocab = 3
        self.n_samples = 2
        self.check_forward_options = {'atol': 1e-4}
        self.check_backward_options = {'atol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-2}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-3, 'rtol': 5e-3}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-3}

    def generate_inputs(self):
        x_shape = (self.batch_size, self.in_size)
        x = numpy.random.uniform(
            -1, 1, x_shape).astype(self.dtype)
        t = numpy.random.randint(
            self.n_vocab, size=self.batch_size).astype(numpy.int32)
        w_shape = (self.n_vocab, self.in_size)
        W = numpy.random.uniform(
            -1, 1, w_shape).astype(self.dtype)
        samples = numpy.random.randint(
            self.n_vocab, size=self.batch_size * self.n_samples) \
            .astype(numpy.int32).reshape((self.batch_size, self.n_samples))
        return x, t, W, samples

    def forward_expected(self, inputs):
        x, t, W, samples = inputs
        expect_y = numpy.empty((self.batch_size), dtype=self.dtype)
        for b in range(self.batch_size):
            z = 0
            for i in range(self.n_samples):
                w = samples[b, i]
                z += numpy.exp(W[w].dot(x[b]))
            y0 = W[t[b]].dot(x[b])
            z += numpy.exp(y0)
            l = y0 - numpy.log(z)
            for i in range(self.n_samples):
                w = samples[b, i]
                l += numpy.log(1 - numpy.exp(W[w].dot(x[b])) / z)

            expect_y[b] = l

        if self.reduce == 'mean':
            loss = -numpy.sum(expect_y) / self.batch_size
        else:
            loss = -expect_y
        return utils.force_array(loss, self.dtype),

    def forward(self, inputs, device):
        x, t, W, samples = inputs
        y = functions.black_out(x, t, W, samples, self.reduce)
        return y,


testing.run_module(__name__, __file__)
